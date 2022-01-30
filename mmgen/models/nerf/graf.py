# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial

import torch
from torch.nn.parallel.distributed import _find_tensors

from mmgen.models import StaticUnconditionalGAN
from mmgen.models.architectures.common import get_module_device
from ..builder import MODELS
from ..common import set_requires_grad
from .nerf import BaseNeRF


@MODELS.register_module()
class GRAF(BaseNeRF, StaticUnconditionalGAN):
    """NOTE: 1. do not use fine render
    """

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 *args,
                 **kwargs):

        BaseNeRF.__init__(self, *args, **kwargs)

        discriminator['input_size'] = self.camera.ray_sampler.N_samples_sqrt
        StaticUnconditionalGAN.__init__(
            self,
            generator=generator,
            discriminator=discriminator,
            gan_loss=gan_loss,
            disc_auxiliary_loss=disc_auxiliary_loss,
            gen_auxiliary_loss=gen_auxiliary_loss)

        self.noise_dim = self.generator.noise_dim
        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        self.train_cfg = dict() if self.train_cfg is None else self.train_cfg

        # set batch chunk for inference to avoid out-of-memory
        self.render_chunk = self.train_cfg.get('render_chunk', 1024 * 32)
        self.network_chunk = self.train_cfg.get('network_chunk', 1024 * 64)

        self.noise_cfg = self.train_cfg.get('noise_cfg', 'uniform')
        self.noise_fn = partial(
            self._add_noise_to_z,
            noise_fn=getattr(self, f'{self.noise_cfg}_noise'))

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        self.test_cfg = dict() if self.test_cfg is None else self.test_cfg

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

    def forward_render(self, batch_size, model, return_noise=False, **kwargs):
        """Forward GRAF's NeRF Generator.

        Returns:
            dict: tensor are all shape like ``[bz, n_points, N]``.
        """

        device = get_module_device(self)
        render_dict = self.camera.prepare_render_rays(
            batch_size=batch_size, device=device)
        camera_pose = render_dict['camera_pose']
        views = render_dict['views']
        rays = render_dict['rays']
        # n_points = rays.shape[0]
        n_points = self.camera.n_points

        results_list = []

        # 1. loop for each render-chunk
        render_chunk = n_points if (
            self.render_chunk is None
            or self.render_chunk == -1) else self.render_chunk

        assert render_chunk % n_points == 0, (
            'To make sure each chunk is contains an integer number of '
            'samples, therefore \'render_chunk\' must be divisible by '
            f'\'n_points\'. But received {n_points} and {render_chunk}')

        batch_chunk = render_chunk // n_points
        for batch_s in range(0, batch_size, batch_chunk):
            batch_e = min(batch_s + batch_chunk, batch_size)

            # 1.1 prepare data for neural_renderer --> slice in batch
            rays_render = rays[batch_s:batch_e]
            views_render = views[batch_s:batch_e]
            camera_pose_render = camera_pose[batch_s:batch_e]
            n_points_chunk = (batch_e - batch_s) * n_points

            # [bz', n_points, n] -> [bz' * n_points, n]
            rays_render = rays_render.reshape([n_points_chunk, -1])
            views_render = views_render.reshape([n_points_chunk, -1])
            views_render = views_render[:, None, :]
            views_render_expand = views_render.expand(-1, self.n_samples, -1)
            camera_pose_render = camera_pose_render.reshape(
                [n_points_chunk, -1])

            # 1.2 sample points
            z_vals_render = self.camera.sample_z_vals(
                self.noise_fn if self.training else None,
                self.n_samples,
                num_points=rays_render.shape[0],  # use the n_points in chunk
                device=device)
            points_render = self.camera.sample_points(rays_render,
                                                      z_vals_render,
                                                      camera_pose_render)

            # generate batch level noise, repeat, and then flatten
            noise_render = torch.randn((batch_e - batch_s), self.noise_dim)
            noise_render = noise_render[:, None, None, :].repeat(
                [1, n_points, self.n_samples, 1])
            noise_render = noise_render.reshape(-1, self.noise_dim).to(device)

            # 1.3 forward network and get raw_output
            network_data_dict = dict(
                views=views_render_expand,
                points=points_render,
                noise=noise_render)

            # update num_batch in current chunk to kwargs
            raw_output = self.forward_network_batchify(network_data_dict,
                                                       self.generator,
                                                       **kwargs)
            # 1.4 volume rendering
            render_results = self.volume_rendering(
                raw_output, z_vals=z_vals_render, ray_vectors=rays_render)

            if return_noise:
                render_results.update(raw_output)
                results_list.append(render_results)
            else:
                # just save render pixels
                results_list.append(render_results['rgb_final'])

        # 2. contentation the results and return
        if return_noise:
            results_dict = self.concatenate_dict(results_list)
            results_dict.update(render_dict)
            results_dict = {
                k: v.reshape(batch_size, n_points, -1)
                for k, v in results_dict.items()
            }

            return results_dict

        return torch.cat(results_list, dim=0).reshape(batch_size, n_points, -1)

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):

        device = get_module_device(self)
        batch_size = data_batch['real_img'].shape[0]

        # get running status
        if running_status is not None:
            self.iteration = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            else:
                self.iteration += 1

        # disc training
        set_requires_grad(self.discriminator, True)
        optimizer['discriminator'].zero_grad()

        # gen fake samples
        with torch.no_grad():
            fake_pixels = self.forward_render(
                batch_size=batch_size, model=self.generator)
        # sample real samples
        real_dict = self.camera.prepare_render_rays(
            real_img=data_batch['real_img'], device=device)
        real_pixels = real_dict['real_pixels']

        print(f'fake_pixels: {fake_pixels.shape}')
        print(f'real_pixels: {real_pixels.shape}')

        import ipdb
        ipdb.set_trace()

        # disc pred for fake imgs and real_imgs
        disc_pred_fake = self.discriminator(fake_pixels)
        disc_pred_real = self.discriminator(real_pixels)
        # get data dict to compute losses for disc
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_pixels,
            # flatten real_imgs to calculate r1-gp for each sample
            real_imgs=real_pixels.reshape(-1, 3),
            iteration=self.iteration,
            batch_size=batch_size,
            loss_scaler=loss_scaler)

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

        if loss_scaler:
            # add support for fp16
            loss_scaler.scale(loss_disc).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(
                    loss_disc, optimizer['discriminator'],
                    loss_id=0) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_disc.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['discriminator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['discriminator'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['discriminator'].step()

        # generator training
        set_requires_grad(self.discriminator, False)
        optimizer['generator'].zero_grad()

        fake_dict = self.forward_render(
            batch_size=batch_size, model=self.generator, return_noise=True)
        disc_pred_fake_g = self.discriminator(fake_dict['rgb_final'])

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            fake_imgs=fake_dict['rgb_final'],
            disc_pred_fake_g=disc_pred_fake_g,
            iteration=self.iteration,
            batch_size=batch_size,
            loss_scaler=loss_scaler)

        loss_gen, log_vars_g = self._get_gen_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))

        if loss_scaler:
            loss_scaler.scale(loss_gen).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(
                    loss_gen, optimizer['generator'],
                    loss_id=1) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_gen.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['generator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['generator'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['generator'].step()

        log_vars = {}
        log_vars.update(log_vars_g)
        log_vars.update(log_vars_disc)

        results = dict(real_imgs=data_batch['real_img'].cpu())
        for res_dict in [real_dict, fake_dict]:
            for k, v in res_dict.items():
                results[k] = v.detach().cpu()
                # print(k, v.shape)
        # import ipdb
        # ipdb.set_trace()

        outputs = dict(
            log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs

    def sample_from_noise(self,
                          noise,
                          num_batches=0,
                          sample_model='ema/orig',
                          **kwargs):
        """Sample images from noises by using the generator.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional):  The number of batch size.
                Defaults to 0.

        Returns:
            torch.Tensor | dict: The output may be the direct synthesized
                images in ``torch.Tensor``. Otherwise, a dict with queried
                data, including generated images, will be returned.
        """
        if sample_model == 'ema':
            assert self.use_ema
            _model = self.generator_ema
        elif sample_model == 'ema/orig' and self.use_ema:
            _model = self.generator_ema
        else:
            _model = self.generator

        outputs = self.forward_render(
            batch_size=num_batches, model=_model, **kwargs)
        # outputs = _model(noise, num_batches=num_batches, **kwargs)

        if isinstance(outputs, dict) and 'noise_batch' in outputs:
            noise = outputs['noise_batch']

        if sample_model == 'ema/orig' and self.use_ema:
            _model = self.generator
            outputs_ = _model(noise, num_batches=num_batches, **kwargs)

            if isinstance(outputs_, dict):
                outputs['rgb_final'] = torch.cat(
                    [outputs['rgb_final'], outputs_['rgb_final']], dim=0)
            else:
                outputs = torch.cat([outputs, outputs_], dim=0)

        return outputs
