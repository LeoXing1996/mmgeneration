# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel.distributed import _find_tensors

from mmgen.models import MODELS, build_module
from mmgen.models.architectures.common import get_module_device
from .camera import Camera
from .util import inverse_transform_sampling


@MODELS.register_module('NeRF')
@MODELS.register_module()
class BaseNeRF(nn.Module, metaclass=ABCMeta):
    """BaseNeRF Module.

    Args:
        camera (dict): config for camera.
    """

    def __init__(self,
                 camera,
                 neural_render,
                 neural_render_fine=None,
                 hierarchical_sampling=None,
                 n_importance=None,
                 num_samples_per_ray=64,
                 white_background=False,
                 nerf_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self._camera_cfg = deepcopy(camera)
        self._neural_renderer_cfg = deepcopy(neural_render)
        self._neural_renderer_fine_cfg = deepcopy(neural_render_fine)

        self.camera = Camera(**camera)
        self.neural_renderer = build_module(neural_render)
        if neural_render_fine is not None and hierarchical_sampling:
            self.neural_renderer_fine = build_module(neural_render_fine)
            assert n_importance is not None and isinstance(
                n_importance, int
            ), ('\'n_importance\' must not be None for hierarchical sampling.')
        self.n_importance = n_importance
        self.n_samples = num_samples_per_ray
        self.white_background = white_background

        if nerf_loss is not None:
            self.nerf_loss = build_module(nerf_loss)
            if not isinstance(self.nerf_loss, nn.ModuleList):
                self.nerf_loss = nn.ModuleList([self.nerf_loss])
        else:
            self.nerf_loss = None
        self.hierarchical_sampling = hierarchical_sampling

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        # buffer to save images in evaluation
        # self._eval_img_buffer = collections.defaultdict(list)
        self._eval_img_buffer = dict()
        self._eval_buffer_iter = -1

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

        self.n_points_train = self.train_cfg.get('num_points_per_image', None)
        self.n_points_eval = None

        self.noise_cfg = self.train_cfg.get('noise_cfg', 'gaussian')
        self.noise_fn = partial(
            self._add_noise_to_z,
            noise_fn=getattr(self, f'{self.noise_cfg}_noise'))

        # set precrop end iter and precrop frac
        self.precrop_iter = self.train_cfg.get('precrop_iter', -1)
        self.precrop_frac = self.train_cfg.get('precrop_frac', None)

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

        # in eval, we sample all points, set n_points as None
        self.n_points_eval = None

        # whether to use exponential moving average for training
        self.use_ema = self.test_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.neural_renderer_ema = deepcopy(self.neural_renderer)
            if hasattr(self, 'neural_renderer_fine'):
                self.neural_renderer_fine_ema = deepcopy(
                    self.neural_renderer_fine)

    def get_precrop_frac(self):
        if not self.training or not hasattr(self, 'precrop_iter'):
            return None
        if self.iteration > self.precrop_iter:
            return None
        return self.precrop_frac

    @property
    def n_points(self):
        return self.n_points_train if self.training else self.n_points_eval

    @property
    def eval_img_buffer(self):
        return self._eval_img_buffer

    @property
    def eval_img_buffer_iter(self):
        return self._eval_buffer_iter

    def collect_img_buffer(self):
        """Collect list to tensor, this function can be called by vis hooks and
        eval hooks.

        This operation is dangerous, maybe we can find a better solution.
        """
        for k, v in self._eval_img_buffer.items():
            if isinstance(v, list):
                self._eval_img_buffer[k] = torch.cat(v, dim=0)
        torch.cuda.empty_cache()

    def _update_img_buffer(self, output_dict):
        """Function for evaluation.

        Different from `self.train_step`, we save the output_dict to the image
        buffer. Then EvalHook and VisHook can load images from this buffer.
        """
        if self._eval_img_buffer:
            assert all([k in output_dict for k in self._eval_img_buffer])
        for k, v in output_dict.items():
            if self.eval_img_buffer_iter == self.iteration:
                # the same iteration, add new element
                self._eval_img_buffer[k].append(v.cpu())
            else:
                # not the same iteration, assign new elements
                self._eval_img_buffer[k] = [v.cpu()]
        self._eval_buffer_iter = self.iteration

    def _get_nerf_loss(self, outputs_dict):
        losses_dict = {}
        for loss_module in self.nerf_loss:
            loss_ = loss_module(outputs_dict)
            if loss_ is None:
                continue
            losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self._parse_losses(losses_dict)
        return loss, log_var

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            # Allow setting None for some loss item.
            # This is to support dynamic loss module, where the loss is
            # calculated with a fixed frequency.
            elif loss_value is None:
                continue
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        # Note that you have to add 'loss' in name of the items that will be
        # included in back propagation.
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def gaussian_noise(self, tar_shape):
        return torch.randn(tar_shape).to(get_module_device(self))

    def _add_noise_to_z(self, z_vals, noise_fn):
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)

        t_rand = noise_fn(list(z_vals.shape))
        z_vals = lower + (upper - lower) * t_rand
        return z_vals

    def get_weight(self, z_vals, ray_vectors, sigma):
        """Calculate weight in Eq TODO
        Args:
            z_vals: [n_points * n_samples]
            ray_vectors: [n_points, 4]
        Returns:
            torch.Tensor: size as [n_points, n_samples]
        """
        deltas = z_vals[..., 1:] - z_vals[..., :-1]
        delta_inf = torch.ones_like(deltas[..., :1]) * 1e10
        deltas = torch.cat([deltas, delta_inf], dim=-1)

        # [n_points, n_sample] * [n_points, 1]
        deltas = deltas * torch.norm(ray_vectors, dim=-1, keepdim=True)
        # alpha = 1. - torch.exp(-F.relu(sigma) * deltas)
        alpha = 1. - torch.exp(-sigma * deltas)

        ones = torch.ones_like(alpha[..., :1])
        weights = alpha * torch.cumprod(
            torch.cat([ones, 1. - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
        return weights

    def volume_rendering(self, raw_output, z_vals, ray_vectors, is_fine=False):
        # NOTE: we should consist the name here
        rgbs, sigmas = raw_output['rgbs'], raw_output['alphas']

        # [n_points, n_samples, 1] to [n_points, n_samples]
        z_vals, sigmas = z_vals.squeeze(), sigmas.squeeze()

        # [n_points, n_samples]
        weights = self.get_weight(z_vals, ray_vectors, sigmas)

        # [bz, n_points, n_samples, 1] * [bz, n_points, n_samples, 3]
        # rgb_final = torch.sum(weights[..., None] * torch.sigmoid(rgbs), -2)
        rgb_map = torch.sum(weights[..., None] * rgbs, -2)
        # [bz, n_points, n_samples] * [bz, n_points, n_samples, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disparity map, which is inversely proportional to the depth
        # [bz, n_points]
        # TODO: here we should use more method to avoid nan than official code
        # in the first around of the training, weights = 0 makes
        # disp_final to none
        disp_map = 1. / torch.max(depth_map / torch.sum(weights, -1),
                                  1e-10 * torch.ones_like(depth_map))
        # [bz, n_points]
        acc_map = torch.sum(weights, -1)

        if self.white_background:
            # NOTE: add comment here, why we calculate this
            rgb_map = rgb_map + (1. - acc_map[..., None])

        suffix = 'final' if (is_fine or not self.hierarchical_sampling) \
            else 'coarse'
        output_dict = {
            f'rgb_{suffix}': rgb_map,
            f'depth_{suffix}': depth_map,
            f'disp_{suffix}': disp_map,
            f'acc_{suffix}': acc_map,
            f'weights_{suffix}': weights
        }
        return output_dict

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
            **data_batch_,
            n_points=self.n_points,
            precrop_frac=self.get_precrop_frac(),
            device=device)
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

    def prepare_hierarchical_sampling(self, z_vals, weights, camera_pose,
                                      rays):

        # z_vals: [1, n_samples] to [n_points, n_samples]
        z_vals = z_vals.expand(weights.shape[0], -1)
        z_vals_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1]).squeeze()
        # z_samples: [n_points, n_samples']
        z_samples = inverse_transform_sampling(
            z_vals_mid,
            weights[..., 1:-1],
            self.n_importance,
            det=not self.training,
            device=get_module_device(self)).detach()

        # z_zals_fine: [n_points, n_samples_fine]
        z_vals_fine, _ = torch.sort(
            torch.cat([z_vals, z_samples], dim=1), dim=1)

        points_fine = self.camera.sample_points(rays, z_vals_fine, camera_pose)
        return points_fine, z_vals_fine

    def forward_network_batchify(self, data_dict, network, **kwargs):
        """Forward the network in chunk."""
        data_keys = list(data_dict.keys())
        # 0. save input shape
        n_points, n_samples, _ = data_dict[data_keys[0]].shape
        # total pixels to render
        n_pixels = n_points * n_samples
        # 1. flatten the input
        data_dict_flatten = {
            k: data_dict[k].reshape(-1, data_dict[k].shape[-1])
            for k in data_keys
        }
        # shape checking
        assert all([
            data_dict_flatten[k].shape[0] == n_pixels for k in data_keys
        ]), ('n_pixels in renderer inputs not consistent.')

        # 2. init list to save output
        raw_output_list = []
        network_chunk = n_pixels if (
            self.network_chunk is None
            or self.network_chunk == -1) else self.network_chunk

        # 3. loop for each mini-batch
        for idx_s in range(0, n_pixels, network_chunk):
            idx_e = idx_s + network_chunk
            mini_data_dict = {
                k: data_dict_flatten[k][idx_s:idx_e]
                for k in data_keys
            }
            raw_output_ = network(**mini_data_dict, **kwargs)
            raw_output_list.append(raw_output_)

        # 4 gather the output dict
        raw_output = self.concatenate_dict(raw_output_list)

        # 5. reshape to [n_points (render_chunk), n_samples, -1]
        for k in raw_output.keys():
            raw_output[k] = raw_output[k].reshape([n_points, n_samples, -1])
        return raw_output

    @staticmethod
    def concatenate_dict(list_of_dict):
        """
        Args:
            list_of_dict ([dict, ]): A list contains dicts to be concatenated.
                All dict have the same keys and values are tensors can be
                concatenated at the first dimension.
        """
        new_dict = dict()
        tar_keys = list_of_dict[0].keys()
        for k in tar_keys:
            new_dict[k] = torch.cat([d[k] for d in list_of_dict], dim=0)
        return new_dict

    def forward_test(self, data, **kwargs):
        """Testing function for NeRF models.

        Args:
            data (torch.Tensor | dict | None): Input data. This data will be
                passed to different methods.
        """
        mode = kwargs.pop('mode', 'sampling')
        if mode in ['sampling', 'reconstruction']:
            # TODO: add random sample from pose
            return self.reconstruction_step(data, **kwargs)

        raise NotImplementedError('Other specific testing functions should'
                                  ' be implemented by the sub-classes.')

    def forward(self, data, return_loss=False, **kwargs):
        """Forward function.

        Args:
            data (dict | torch.Tensor): Input data dictionary.
            return_loss (bool, optional): Whether in training or testing.
                Defaults to False.

        Returns:
            dict: Output dictionary.
        """
        if return_loss:
            return self.forward_train(data, **kwargs)

        return self.forward_test(data, **kwargs)
