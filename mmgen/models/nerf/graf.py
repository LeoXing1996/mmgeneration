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

# import sys

# sys.path.append('/space0/home/xingzn/code/nerf/graf/')
# sys.path.append('/space0/home/xingzn/code/nerf/graf/submodules')


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

        # from graf.models.discriminator import Discriminator
        # self.discriminator = Discriminator(imsize=32).cuda()
        # self.nerf_off_kwargs = self.load_official_generator()
        # self.nerf_off_kwargs[0]['network_fn']
        # from torch.optim import RMSprop
        # self.gen_opt = RMSprop(
        #     self.nerf_off_kwargs[0]['network_fn'].parameters(),
        #     lr=5e-4,
        #     eps=1e-8,
        #     weight_decay=0,
        #     momentum=0,
        #     alpha=0.99)

        # state_dict = torch.load('/space0/home/xingzn/mmgen_dev/nerf/'
        #                         'work_dirs/ckpts/GRAF-weights/'
        #                         'carla_128_cvt.pt', map_location='cpu')
        # # ignore disc
        # state_dict = {k: v for k, v in state_dict.items() if 'disc' not in k}
        # self.load_state_dict(state_dict, strict=False)

    # def load_official_generator(self):
    #     from GAN_stability.gan_training import config
    #     from nerf_pytorch import run_nerf_mod
    #     import os.path as op
    #     from argparse import Namespace

    #     config_root = '/space0/home/xingzn/code/nerf/graf/configs'
    #     dataset = 'carla_128'
    #     config_dict = dict(
    #         carla_128='carla.yaml',
    #         carla_256='carla_256.yaml',
    #         carla_512='carla_512.yaml')
    #     config_path = op.join(config_root, config_dict[dataset])
    #     # 1. load default
    #     config_dict = config.load_config(config_path,
    #                                      op.join(config_root, 'default.yaml'))  # noqa

    #     # 2. overwrite
    #     config_nerf = Namespace(**config_dict['nerf'])
    #     # Update config for NERF
    #     config_nerf.chunk = min(
    #         config_dict['training']['chunk'],
    #         1024 * config_dict['training']['batch_size']
    #     )  # let batch size for training with patches limit the maximal memory # noqa
    #     config_nerf.netchunk = config_dict['training']['netchunk']
    #     config_nerf.white_bkgd = config_dict['data']['white_bkgd']
    #     config_nerf.feat_dim = config_dict['z_dist']['dim']
    #     config_nerf.feat_dim_appearance = config_dict['z_dist'][
    #         'dim_appearance']

    #     nerf_off_args = run_nerf_mod.create_nerf(config_nerf)

    #     return nerf_off_args

    # def input_to_official(self, batch_size, device):
    #     render_dict = self.camera.prepare_render_rays(
    #         batch_size=batch_size, device=device)

    #     # [batch_size, n_points, 3/4]
    #     pose, views = render_dict['camera_pose'], render_dict['views']
    #     pose = pose[..., :3].reshape(-1, 3)
    #     views = views[..., :3].reshape(-1, 3)
    #     rays_official = torch.cat([pose[None, ], views[None, ]], dim=0)
    #     return rays_official

    # def forward_official_render(self, batch_size, device, return_noise=False): # noqa
    #     from nerf_pytorch.run_nerf_mod import render
    #     nerf_kwargs = self.nerf_off_kwargs[0]

    #     rays = self.input_to_official(batch_size, device)
    #     noise = torch.randn(batch_size, 256).to(device)
    #     nerf_kwargs['features'] = noise
    #     rgb, disp, acc, extras = render(
    #         128,
    #         128,
    #         self.camera.focal,
    #         chunk=self.render_chunk,
    #         rays=rays,
    #         near=self.camera.near,
    #         far=self.camera.far,
    #         **nerf_kwargs)

    #     def rays_to_output(x):
    #         return x.view(len(x), -1) * 2 - 1

    #     rgb = rays_to_output(rgb)
    #     disp = rays_to_output(disp)
    #     acc = rays_to_output(acc)

    #     if return_noise:
    #         output_dict = dict(rgb_final=rgb, disp_final=disp, acc_final=acc)
    #         return output_dict
    #     return rgb

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

        # whether to use exponential moving average for training
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.generator_ema = deepcopy(self.generator)

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
            self.generator_ema = deepcopy(self.generator)

    def forward_render(self,
                       batch_size,
                       model,
                       noise=None,
                       return_noise=False,
                       **kwargs):
        """Forward GRAF's NeRF Generator.

        Args:
            noise (torch.Tensor)

        Returns:
            dict: tensor are all shape like ``[bz, n_points, N]``.
        """

        device = get_module_device(self)
        kwargs['return_noise'] = return_noise
        render_dict = self.camera.prepare_render_rays(
            batch_size=batch_size, device=device)
        camera_pose = render_dict['camera_pose']
        views = render_dict['views']
        rays = render_dict['rays']
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
                                                       model, **kwargs)
            # 1.4 volume rendering
            render_results = self.volume_rendering(
                raw_output, z_vals=z_vals_render, ray_vectors=rays_render)

            # NOTE: rerange rgb, disp and acc to [-1, 1], same as official ones
            for k in render_results:
                if any([norm_kw in k for norm_kw in ['rgb', 'acc', 'disp']]):
                    render_results[k] = render_results[k] * 2 - 1

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

        # update noise in generator
        self.update_noise(self.generator, self.iteration)

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

        # behavior of NoisyReLU is different train between evaluation modes
        _model.eval()
        outputs = self.forward_render(
            batch_size=num_batches, model=_model, **kwargs)
        _model.train()

        if isinstance(outputs, dict) and 'noise_batch' in outputs:
            noise = outputs['noise_batch']

        if sample_model == 'ema/orig' and self.use_ema:
            # behavior of NoisyReLU is different train between evaluation modes
            _model.eval()
            _model = self.generator
            outputs_ = self.forward_render(
                batch_size=num_batches, model=_model, noise=noise, **kwargs)
            _model.train()

            if isinstance(outputs_, dict):
                outputs['rgb_final'] = torch.cat(
                    [outputs['rgb_final'], outputs_['rgb_final']], dim=0)
            else:
                outputs = torch.cat([outputs, outputs_], dim=0)

        return outputs
