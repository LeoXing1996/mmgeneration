# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import _find_tensors

from mmgen.models import MODELS, build_module
from mmgen.models.architectures.common import get_module_device
from .base_nerf import BaseNeRF


@MODELS.register_module()
class NeRF(BaseNeRF):
    """BaseNeRF Module.

    Args:
        camera (dict): config for camera.
    """

    def __init__(self,
                 neural_render,
                 neural_render_fine=None,
                 nerf_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._neural_renderer_cfg = deepcopy(neural_render)
        self._neural_renderer_fine_cfg = deepcopy(neural_render_fine)

        self.neural_renderer = build_module(neural_render)
        if neural_render_fine is not None and self.hierarchical_sampling:
            self.neural_renderer_fine = build_module(neural_render_fine)
            assert self.n_importance is not None and isinstance(
                self.n_importance, int
            ), ('\'n_importance\' must not be None for hierarchical sampling.')

        if nerf_loss is not None:
            self.nerf_loss = build_module(nerf_loss)
            if not isinstance(self.nerf_loss, nn.ModuleList):
                self.nerf_loss = nn.ModuleList([self.nerf_loss])
        else:
            self.nerf_loss = None

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_eval_cfg()

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()

        # set batch chunk for inference to avoid out-of-memory
        self.render_chunk = self.train_cfg.get('render_chunk', 1024 * 32)
        self.network_chunk = self.train_cfg.get('network_chunk', 1024 * 64)

        self.noise_cfg = self.train_cfg.get('noise_cfg', 'uniform')
        self.noise_fn = partial(
            self._add_noise_to_z,
            noise_fn=getattr(self, f'{self.noise_cfg}_noise'))

        # whether to use exponential moving average for training
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.neural_renderer_ema = deepcopy(self.neural_renderer)
            if hasattr(self, 'neural_renderer_fine'):
                self.neural_renderer_fine_ema = deepcopy(
                    self.neural_renderer_fine)

    def _parse_eval_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # set batch chunk for inference to avoid out-of-memory
        self.render_chunk = self.test_cfg.get('render_chunk', 1024 * 32)
        self.network_chunk = self.test_cfg.get('network_chunk', 1024 * 64)

        # whether to use exponential moving average for training
        self.use_ema = self.test_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.neural_renderer_ema = deepcopy(self.neural_renderer)
            if hasattr(self, 'neural_renderer_fine'):
                self.neural_renderer_fine_ema = deepcopy(
                    self.neural_renderer_fine)

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):

        optimizer.zero_grad()
        batch_size, H, W, _ = data_batch['real_img'].shape
        assert batch_size == 1, 'We only support batch_size=1 currently'

        # get running status
        if running_status is not None:
            self.iteration = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            else:
                self.iteration += 1

        output_dict = self.reconstruction_step(data_batch, return_noise=True)
        loss, log_vars = self._get_nerf_loss(output_dict)

        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss))
        if loss_scaler:
            # add support for fp16
            loss_scaler.scale(loss).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer)
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer)
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer.step()

        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=output_dict)

        return outputs

    def reconstruction_step(self,
                            data_batch,
                            return_noise=False,
                            sample_model='ema/orig',
                            **kwargs):
        """
        Args:
            data_batch: To be noted that batch_size dimension has been
                squeezed.
            return_noise: Do not have realistic meaning, just a flag to denote
                the network is traininig. If True, denotes this function is
                used for training. Defaults to False.
        """
        # TODO: handle sample-model ->
        # we need an additional function to forward

        # check batch size is 1
        assert all([
            v.shape[0] == 1 for v in data_batch.values()
            if isinstance(v, torch.Tensor)
        ]), ('Batch size of the input \'data_batch\' must be 1.')
        # squeeze the data_batch_
        data_batch_ = {
            k: v.squeeze(0)
            for k, v in data_batch.items() if isinstance(v, torch.Tensor)
        }

        device = get_module_device(self)
        # 0. prepare rays, points, and other variables for rendering
        # TODO: here we can make a data_mapping like Loss
        render_dict = self.camera.prepare_render_rays(
            **data_batch_, device=device)
        camera_pose = render_dict['camera_pose']
        views = render_dict['views']
        rays = render_dict['rays']

        # get n_points from plane to avoid self.n_points is None
        n_points = rays.shape[0]

        # init an empty results dict
        results_list = []

        # 1. loop for each render-chunk
        render_chunk = n_points if (
            self.render_chunk is None
            or self.render_chunk == -1) else self.render_chunk

        for render_s in range(0, n_points, render_chunk):
            render_e = render_s + render_chunk
            # 1.1 prepare data for neural_renderer --> slice
            rays_render = rays[render_s:render_e]
            views_render = views[render_s:render_e]
            views_render = views_render[:, None, :]
            views_render_expand = views_render.expand(-1, self.n_samples, -1)
            # 1.2 sample points
            z_vals_render = self.camera.sample_z_vals(
                self.noise_fn if self.training else None,
                self.n_samples,
                num_points=rays_render.shape[0],
                device=device)
            points_render = self.camera.sample_points(rays_render,
                                                      z_vals_render,
                                                      camera_pose)
            # 1.3 forward network and get raw_output
            network_data_dict = dict(
                views=views_render_expand, points=points_render)
            raw_output = self.forward_network_batchify(network_data_dict,
                                                       self.neural_renderer,
                                                       **kwargs)
            # 1.4 volume rendering
            render_results = self.volume_rendering(
                raw_output, z_vals=z_vals_render, ray_vectors=rays_render)

            # 1.5 hierarchial sampling
            if self.hierarchical_sampling:
                # 1.5.1 update sample points
                weights_render = render_results['weights_coarse']
                points_fine, z_vals_fine = self.prepare_hierarchical_sampling(
                    z_vals_render, weights_render, camera_pose, rays_render)

                n_samples_fine = points_fine.shape[1]
                views_render_expand = views_render.expand(
                    -1, n_samples_fine, -1)
                # 1.5.2 forward network_fine
                network_data_dict_fine = dict(
                    views=views_render_expand, points=points_fine)
                raw_output_fine = self.forward_network_batchify(
                    network_data_dict_fine, self.neural_renderer_fine,
                    **kwargs)
                # 1.5.3 volume render fine
                render_results_fine = self.volume_rendering(
                    raw_output_fine,
                    z_vals=z_vals_fine,
                    ray_vectors=rays_render,
                    is_fine=True)

                render_results.update(render_results_fine)

            if return_noise:
                render_results.update(raw_output)
                results_list.append(render_results)
            else:
                # just save render pixels
                results_list.append(render_results['rgb_final'])

        # 2. cat the results and return
        if return_noise:
            results_dict = self.concatenate_dict(results_list)
            results_dict.update(render_dict)
            # TODO: instance checking can be removed later, for now in use
            # some np array for debug
            results_dict = {
                k: v.unsqueeze(0)
                for k, v in results_dict.items()
                if isinstance(v, torch.Tensor)
            }

            # print(self.iteration, self.training)
            if not self.training:
                self._update_img_buffer(results_dict)
            return results_dict

        return torch.cat(results_list, dim=0).unsqueeze(0)
