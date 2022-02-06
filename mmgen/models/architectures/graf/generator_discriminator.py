# Copyright (c) OpenMMLab. All rights reserved.
import logging

import mmcv
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmcv.runner.checkpoint import _load_checkpoint_with_prefix

from mmgen.models.architectures.common import get_module_device
from mmgen.models.architectures.nerf import NeRFRenderer
from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger
from .modules import GRAFDiscBlock


@MODULES.register_module()
class GRAFGenerator(nn.Module):
    """Generator for GRAF. In this class, we parse GRAF's input to NeRF ones
    and use the off-the-shelf NeRF's generator.

    Args:
        shape_dim
        app_dim
    """

    def __init__(self,
                 shape_dim=128,
                 app_dim=128,
                 init_type=None,
                 *args,
                 **kwargs):
        super().__init__()
        # recalculate the input channels
        if shape_dim is not None:
            assert isinstance(
                shape_dim,
                int), ('\'shape_dim\' must be int or None, but recieive '
                       f'\'{shape_dim}\'')
            self.shape_dim = shape_dim
        else:
            self.shape_dim = 0

        if app_dim is not None:
            assert isinstance(
                app_dim,
                int), ('\'app_dim\' must be int or None, but recieive '
                       f'\'{shape_dim}\'')
            self.app_dim = app_dim
        else:
            self.app_dim = 0

        self.noise_dim = self.shape_dim + self.app_dim

        self.model = NeRFRenderer(
            input_ch_add=shape_dim,
            input_ch_views_add=app_dim,
            *args,
            **kwargs)

        self.init_type = init_type
        self._warning_raised = False

    def forward(self, points, views=None, noise=None, return_noise=False):
        """Forward function.

        Args:
            points (torch.Tensor): Shape as [n_points', n_samples, 4]
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler. Defaults
                to None.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            views (torch.Tensor, optional): Shape as [n_points', n_samples, 4].
                Defaults to None.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.

        Returns:
            torch.Tensor | dict: If not ``return_noise``, only the output
                image will be returned. Otherwise, a dict contains
                ``fake_image``, ``noise_batch`` and ``label_batch``
                would be returned.
        """
        target_shape = (points.shape[0], self.shape_dim + self.app_dim)
        if isinstance(noise, torch.Tensor):
            noise_batch = noise
        else:
            # receive a noise generator and sample noise.
            if callable(noise):
                noise_generator = noise
                noise_batch = noise_generator(target_shape)
            # otherwise, we will adopt default noise sampler.
            else:
                noise_batch = torch.randn(target_shape)
            if not self._warning_raised:
                mmcv.print_log(
                    f'Generate random noise shape as {target_shape}, this '
                    'operation may cause points belong to the same sample '
                    'use the different latent code. Please consider to '
                    'generate noise in the outer models.', 'mmcv',
                    logging.WARNING)

        noise_batch = noise_batch.to(get_module_device(self))

        # noise shape checking
        assert noise_batch.shape == target_shape, (
            f'The noise should be in shape of {target_shape}, but got '
            f'{noise.shape}')

        # 1. split noise_batch to shape and appearance (if have)
        shape_feature, app_feature = noise_batch.split(
            [self.shape_dim, self.app_dim], dim=1)

        # 2. forward nerf (self.model)
        output_dict = self.model(points, views, shape_feature, app_feature)

        # 3. handle return noise
        if return_noise:
            output_dict['noise_batch'] = noise_batch
        return output_dict

    def init_weights(self, pretrained=None):
        """Init weights for GRAF's generator. Because GRAF share the same
        generator with NeRF, we direct use NeRF's weight initialization
        function.

        Args:
            pretrained (str | dict, optional): Path for the pretrained model or
                dict containing information for pretained models whose
                necessary key is 'ckpt_path'. Besides, you can also provide
                'prefix' to load the generator part from the whole state dict.
                Defaults to None.
        """
        # use pytorch's default init method
        if (self.init_type is not None and pretrained is not None
                and self.init_type.upper() == 'DEFAULT'):
            return
        # otherwise call NeRF's init method or load from pretrained model
        self.model.init_weights(pretrained)


@MODULES.register_module()
class GRAFDiscriminator(nn.Module):
    """Discriminator for GRAF.

    Args:
    """

    default_channel_list = {
        128: [0.5, 2, 2, 2, 2, 1],
        64: [1, 2, 4, 8],
        32: [2, 4, 8]
    }

    default_IN_factor = {128: [1, 2, 3], 64: [1, 2, 3], 32: [0, 1, 2]}

    default_activate = dict(
        type='LeakyReLU', negative_slope=0.2, inplace=False)

    def __init__(self,
                 input_size,
                 input_channels,
                 base_channels,
                 inplace_relu=False,
                 pretrained=None):

        super().__init__()
        self.input_size = input_size
        self.input_channels = input_channels

        channel_list = self.default_channel_list[input_size]
        IN_list = self.default_IN_factor[input_size]
        act_cfg = self.default_activate
        act_cfg['inplace'] = inplace_relu

        blocks = []
        for idx in range(len(channel_list)):
            # use `int()` in channels calculation because we have 0.5
            # in channel_list
            in_chn = input_channels if idx == 0 else \
                int(channel_list[idx-1] * base_channels)
            out_chn = int(channel_list[idx] * base_channels)
            use_IN = idx in IN_list
            blocks.append(
                GRAFDiscBlock(in_chn, out_chn, 4, 2, 1, act_cfg, use_IN))
        blocks.append(GRAFDiscBlock(out_chn, 1, 4, 1, 0, None))

        self.blocks = nn.Sequential(*blocks)
        self.init_weights(pretrained)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape like [bz*inp_size*inp_size, 3]
        """
        x = x.view(-1, self.input_size, self.input_size, self.input_channels)
        x = x.permute(0, 3, 1, 2)

        for conv_block in self.blocks:
            x = conv_block(x)

        return x

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str | dict, optional): Path for the pretrained model or
                dict containing information for pretained models whose
                necessary key is 'ckpt_path'. Besides, you can also provide
                'prefix' to load the generator part from the whole state dict.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif isinstance(pretrained, dict):
            ckpt_path = pretrained.get('ckpt_path', None)
            assert ckpt_path is not None
            prefix = pretrained.get('prefix', '')
            map_location = pretrained.get('map_location', 'cpu')
            strict = pretrained.get('strict', True)
            state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                      map_location)
            self.load_state_dict(state_dict, strict=strict)
            mmcv.print_log(f'Load pretrained model from {ckpt_path}', 'mmgen')
        else:
            # just use default init method
            assert pretrained is None, ('pretrained must be a str or None but'
                                        f' got {type(pretrained)} instead.')
