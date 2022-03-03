# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import _find_tensors

from ..builder import MODELS, build_module
from ..common import set_requires_grad
from .base_gan import BaseGAN


@MODELS.register_module()
class CIPS3D(BaseGAN):
    """Class for CIPS-3D.

    Args:
        generator (dict): Config for generator.
        discriminator (dict): Config for discriminator.
        gan_loss (dict): Config for generative adversarial loss.
        disc_auxiliary_loss (dict): Config for auxiliary loss to
            discriminator.
        gen_auxiliary_loss (dict | None, optional): Config for auxiliary loss
            to generator. Defaults to None.
        train_cfg (dict | None, optional): Config for training schedule.
            Defaults to None.
        test_cfg (dict | None, optional): Config for testing schedule. Defaults
            to None.
        num_classes (int | None, optional): The number of conditional classes.
            Defaults to None.
    """

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self._gen_cfg = deepcopy(generator)
        self.G_kwargs = self._gen_cfg.pop('kwargs')
        self.generator = build_module(self._gen_cfg)

        # support no discriminator in testing
        if discriminator is not None:
            self._disc_cfg = deepcopy(discriminator)
            self.discriminator = build_module(discriminator)
        else:
            self.discriminator = None

        # support no gan_loss in testing
        if gan_loss is not None:
            self.gan_loss = build_module(gan_loss)
        else:
            self.gan_loss = None

        if disc_auxiliary_loss:
            self.disc_auxiliary_losses = build_module(disc_auxiliary_loss)
            if not isinstance(self.disc_auxiliary_losses, nn.ModuleList):
                self.disc_auxiliary_losses = nn.ModuleList(
                    [self.disc_auxiliary_losses])
        else:
            self.disc_auxiliary_loss = None

        if gen_auxiliary_loss:
            self.gen_auxiliary_losses = build_module(gen_auxiliary_loss)
            if not isinstance(self.gen_auxiliary_losses, nn.ModuleList):
                self.gen_auxiliary_losses = nn.ModuleList(
                    [self.gen_auxiliary_losses])
        else:
            self.gen_auxiliary_losses = None

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)
        self.gen_steps = self.train_cfg.get('gen_steps', 1)

        # add support for accumulating gradients within multiple steps. This
        # feature aims to simulate large `batch_sizes` (but may have some
        # detailed differences in BN). Note that `self.disc_steps` should be
        # set according to the batch accumulation strategy.
        # In addition, in the detailed implementation, there is a difference
        # between the batch accumulation in the generator and discriminator.
        self.batch_accumulation_steps = self.train_cfg.get(
            'batch_accumulation_steps', 1)

        # whether to use exponential moving average for training
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # use deepcopy to guarantee the consistency
            self.generator_ema = deepcopy(self.generator)

        self.img_size = self.train_cfg.get('img_size', None)

        self.nerf_noise_disable = not self.train_cfg.get('nerf_noise', True)
        self.warmup_alpha = self.train_cfg.get('warmup_D', False)
        self.fade_step = self.train_cfg.get('fade_step', 10000)

        self.points_per_forward = self.train_cfg.get('points_per_forward', 256)
        self.grad_points_per_forward = self.train_cfg.get('grad_points', 256)

        # aux discriminator for nerf output
        self.aux_disc = self.train_cfg.get('aux_disc', True)
        self.aux_disc_freq = self.train_cfg.get('aux_disc_freq', 1)

        self.use_fp16 = self.train_cfg.get('use_fp16', False)

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)

        if self.img_size is None:
            self.img_size = self.test_cfg.get('img_size', None)

    @property
    def nerf_noise(self):
        """Get noise used in nerf process."""
        if self.nerf_noise_disable:
            return 0
        else:
            return max(0, 1.0 - self.iteration / 5000)

    @property
    def alpha(self):
        """Get warmup alpha for discriminator."""
        if self.warmup_alpha:
            return min(1, self.iteration / self.fade_step)
        else:
            return 1

    @property
    def aux_reg(self):
        """Whether use aux discriminator or not."""
        if self.aux_disc and self.iteration % self.aux_disc_freq == 0:
            return True
        return False

    @property
    def forward_points(self):
        """How many points are generated in one forward."""
        if self.img_size >= 256 and self.points_per_forward is not None:
            return self.points_per_forward**2
        return None

    @property
    def grad_points(self):
        """How many poitns are used to calculate grad."""
        if self.grad_points_per_forward is not None:
            return self.grad_points_per_forward
        return None

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):
        """Train step function.

        This function implements the standard training iteration for
        asynchronous adversarial training. Namely, in each iteration, we first
        update discriminator and then compute loss for generator with the newly
        updated discriminator.

        As for distributed training, we use the ``reducer`` from ddp to
        synchronize the necessary params in current computational graph.

        Args:
            data_batch (dict): Input data from dataloader.
            optimizer (dict): Dict contains optimizer for generator and
                discriminator.
            ddp_reducer (:obj:`Reducer` | None, optional): Reducer from ddp.
                It is used to prepare for ``backward()`` in ddp. Defaults to
                None.
            loss_scaler (:obj:`torch.cuda.amp.GradScaler` | None, optional):
                The loss/gradient scaler used for auto mixed-precision
                training. Defaults to ``None``.
            use_apex_amp (bool, optional). Whether to use apex.amp. Defaults to
                ``False``.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Contains 'log_vars', 'num_samples', and 'results'.
        """
        # get data from data_batch
        real_imgs = data_batch['real_img']
        # If you adopt ddp, this batch size is local batch size for each GPU.
        # If you adopt dp, this batch size is the global batch size as usual.
        batch_size = real_imgs.shape[0]
        if self.img_size is None:
            self.img_size = real_imgs.shape[-1]
        else:
            assert self.img_size == real_imgs.shape[-1]

        # get running status
        if running_status is not None:
            self.iteration = curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # disc training
        set_requires_grad(self.discriminator, True)

        # do not `zero_grad` during batch accumulation
        if curr_iter % self.batch_accumulation_steps == 0:
            optimizer['discriminator'].zero_grad()

        with torch.cuda.amp.autocast(self.use_fp16):
            with torch.no_grad():
                # zs_list = self.generator.module.get_zs(real_imgs.shape[0])
                zs_list = self.generator.get_zs(real_imgs.shape[0])
                fake_imgs, fake_pos = self.generator(
                    zs_list,
                    img_size=self.img_size,
                    nerf_noise=self.nerf_noise,
                    return_aux_img=self.aux_reg,
                    forward_points=self.forward_points,
                    grad_points=None,
                    hierarchical_sample=False,
                    **self.G_kwargs)
                # fake_imgs: [img, img_nerf]
            if self.aux_reg:
                # one for disc and one for aux disc
                real_imgs = torch.cat([real_imgs, real_imgs], dim=0)
                # real_imgs.requires_grad_()

        # disc pred for fake imgs and real_imgs
        disc_pred_real = self.discriminator(
            real_imgs,
            alpha=self.alpha,
            use_aux_disc=self.aux_reg,
            return_latent=False)
        disc_pred_fake = self.discriminator(
            real_imgs,
            alpha=self.alpha,
            use_aux_disc=self.aux_reg,
            return_latent=False)

        # get data dict to compute losses for disc
        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size,
            loss_scaler=loss_scaler)

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)
        loss_disc = loss_disc / float(self.batch_accumulation_steps)

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

        if (curr_iter + 1) % self.batch_accumulation_steps == 0:
            if loss_scaler:
                loss_scaler.unscale_(optimizer['discriminator'])
                # note that we do not contain clip_grad procedure
                loss_scaler.step(optimizer['discriminator'])
                # loss_scaler.update will be called in runner.train()
            else:
                optimizer['discriminator'].step()

        # skip generator training if only train discriminator for current
        # iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            results = dict(
                fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
            outputs = dict(
                log_vars=log_vars_disc,
                num_samples=batch_size,
                results=results)
            if hasattr(self, 'iteration'):
                self.iteration += 1
            return outputs

        # generator training
        set_requires_grad(self.discriminator, False)
        # allow for training the generator with multiple steps
        for _ in range(self.gen_steps):
            optimizer['generator'].zero_grad()
            for _ in range(self.batch_accumulation_steps):
                # zs_list = self.generator.module.get_zs(real_imgs.shape[0])
                zs_list = self.generator.get_zs(real_imgs.shape[0])

                with torch.cuda.amp.autocast(self.use_fp16):
                    fake_imgs, fake_pos = self.generator(
                        zs_list,
                        img_size=self.img_size,
                        nerf_noise=self.nerf_noise,
                        return_aux_img=self.aux_reg,
                        forward_points=None,
                        grad_points=None,
                        hierarchical_sample=False,
                        **self.G_kwargs)

                disc_pred_fake_g = self.discriminator(
                    fake_imgs.to(torch.float32),
                    alpha=self.alpha,
                    use_aux_disc=self.aux_reg,
                    return_latent=False)

                data_dict_ = dict(
                    gen=self.generator,
                    disc=self.discriminator,
                    fake_imgs=fake_imgs,
                    disc_pred_fake_g=disc_pred_fake_g,
                    iteration=curr_iter,
                    batch_size=batch_size,
                    loss_scaler=loss_scaler)

                loss_gen, log_vars_g = self._get_gen_loss(data_dict_)
                loss_gen = loss_gen / float(self.batch_accumulation_steps)

                # prepare for backward in ddp. If you do not call this function
                # before back propagation, the ddp will not dynamically find
                # the used params in current computation.
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

        results = dict(fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
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
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            sampel_model (str, optional): Use which model to sample fake
                images. Defaults to `'ema/orig'`.
            label (torch.Tensor | None , optional): The conditional label.
                Defaults to None.

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

        g_kwargs_ = deepcopy(self.G_kwargs)
        g_kwargs_['img_size'] = self.img_size
        g_kwargs_['psi'] = 1

        zs_list = _model.module.get_zs(batch_size=num_batches)
        _model.eval()
        outputs = _model(zs_list, forward_points=256**2, **g_kwargs_)[0]
        _model.train()
        # outputs = _model(noise, num_batches=num_batches,
        #                  label=label, **kwargs)

        if sample_model == 'ema/orig' and self.use_ema:
            _model = self.generator
            _model.eval()
            outputs_ = _model(zs_list, forward_poinst=256**2, *g_kwargs_)[0]
            _model.train()

            if isinstance(outputs_, dict):
                outputs_['fake_img'] = torch.cat(
                    [outputs['fake_img'], outputs_['fake_img']], dim=0)
            else:
                outputs = torch.cat([outputs, outputs_], dim=0)

        return outputs
