# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmgen.models.builder import MODULES
from .diffaug import DiffAugment
from .module import (ConvLayer, EqualLinear, LinearBlock, ResBlock, SinBlock,
                     ToRGB, frequency_init)


@MODULES.register_module('CIPS3DGenerator')
class CIPSGenerator(nn):

    def __init__(self,
                 input_dim,
                 style_dim,
                 hidden_dim=256,
                 pre_rgb_dim=32,
                 device=None,
                 name_prefix='inr',
                 **kwargs):
        """

        :param input_dim:
        :param style_dim:
        :param hidden_dim:
        :param pre_rgb_dim:
        :param device:
        :param name_prefix:
        :param kwargs:
        """
        super().__init__()

        self.device = device
        self.pre_rgb_dim = pre_rgb_dim
        self.name_prefix = name_prefix

        self.channels = {
            '4': hidden_dim,
            '8': hidden_dim,
            '16': hidden_dim,
            '32': hidden_dim,
            '64': hidden_dim,
            '128': hidden_dim,
            '256': hidden_dim,
            '512': hidden_dim,
            '1024': hidden_dim,
        }

        self.module_name_list = []

        self.style_dim_dict = {}

        _out_dim = input_dim

        network = OrderedDict()
        to_rbgs = OrderedDict()
        for i, (name, channel) in enumerate(self.channels.items()):
            _in_dim = _out_dim
            _out_dim = channel

            if name.startswith(('none', )):
                _linear_block = LinearBlock(
                    in_dim=_in_dim,
                    out_dim=_out_dim,
                    name_prefix=f'{name_prefix}_{name}')
                network[name] = _linear_block
            else:
                _film_block = SinBlock(
                    in_dim=_in_dim,
                    out_dim=_out_dim,
                    style_dim=style_dim,
                    name_prefix=f'{name_prefix}_w{name}')
                self.style_dim_dict.update(_film_block.style_dim_dict)
                network[name] = _film_block

            _to_rgb = ToRGB(
                in_dim=_out_dim, dim_rgb=pre_rgb_dim, use_equal_fc=False)
            to_rbgs[name] = _to_rgb

        self.network = nn.ModuleDict(network)
        self.to_rgbs = nn.ModuleDict(to_rbgs)
        self.to_rgbs.apply(frequency_init(100))
        self.module_name_list.append('network')
        self.module_name_list.append('to_rgbs')

        out_layers = []
        if pre_rgb_dim > 3:
            out_layers.append(nn.Linear(pre_rgb_dim, 3))
        out_layers.append(nn.Tanh())
        self.tanh = nn.Sequential(*out_layers)
        # self.tanh.apply(init_func.kaiming_leaky_init)
        self.tanh.apply(frequency_init(100))
        self.module_name_list.append('tanh')

        models_dict = {}
        for name in self.module_name_list:
            models_dict[name] = getattr(self, name)
        models_dict['cips'] = self

    def forward_orig(self, input, style_dict, img_size=1024, **kwargs):
        """

        :param input: points xyz, (b, num_points, 3)
        :param style_dict:
        :param ray_directions: (b, num_points, 3)
        :param kwargs:
        :return:
        - out: (b, num_points, 4), rgb(3) + sigma(1)
        """

        x = input
        img_size = str(2**int(np.log2(img_size)))

        rgb = 0
        for idx, (name, block) in enumerate(self.network.items()):
            # skip = int(name) >= 32
            if idx >= 4:
                skip = True
            else:
                skip = False
            x = block(x, style_dict, skip=skip)

            if idx >= 3:
                rgb = self.to_rgbs[name](x, skip=rgb)

            if name == img_size:
                break

        out = self.tanh(rgb)
        return out

    def forward(self,
                noise,
                style,
                return_noise=False,
                return_label=False,
                **kwargs):
        """This function implement a forward function in mmgen's style.

        Args:
            noise: The input feature.
            style: The style dict.

        Returns:
            output: Tensor shape as (bz, n_points, 4) or dict.
        """
        output = self.forward_orig(noise, style)

        if return_noise:
            output_dict = dict(
                fake_pixels=output, style=style, noise_batch=noise)
            return output_dict
        return output


class MultiScaleDiscriminator(nn.Module):

    def __init__(self,
                 diffaug,
                 max_size,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 input_size=3,
                 first_downsample=False,
                 channels=None,
                 stddev_group=4,
                 **kwargs):
        super().__init__()

        self.epoch = 0
        self.step = 0

        self.diffaug = diffaug
        self.max_size = max_size
        self.input_size = input_size
        self.stddev_group = stddev_group

        self.module_name_list = []

        if channels is None:
            channels = {
                4: 512,
                8: 512,
                16: 512,
                32: 512,
                64: 256 * channel_multiplier,
                128: 128 * channel_multiplier,
                256: 64 * channel_multiplier,
                512: 32 * channel_multiplier,
                1024: 16 * channel_multiplier,
            }

        self.conv_in = nn.ModuleDict()
        self.module_name_list.append('conv_in')
        for name, channel_ in channels.items():
            self.conv_in[f'{name}'] = ConvLayer(input_size, channel_, 1)

        self.convs = nn.ModuleDict()
        self.module_name_list.append('convs')
        log_size = int(math.log(max_size, 2))
        in_channel = channels[max_size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i - 1)]
            self.convs[f'{2 ** i}'] = ResBlock(
                in_channel,
                out_channel,
                blur_kernel,
                first_downsample=first_downsample)
            in_channel = out_channel

        self.stddev_feat = 1

        if self.stddev_group > 1:
            self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        else:
            self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.module_name_list.append('final_conv')

        self.space_linear = EqualLinear(
            channels[4] * 4 * 4, channels[4], activation='fused_lrelu')
        self.module_name_list.append('space_linear')

        self.out_linear = EqualLinear(channels[4], 1)
        self.module_name_list.append('out_linear')

        models_dict = {}
        for name in self.module_name_list:
            models_dict[name] = getattr(self, name)
        models_dict['D'] = self

    def diff_aug_img(self, img):
        img = DiffAugment(img, policy='color,translation,cutout')
        return img

    def forward(self, input, alpha, summary_ddict=None):
        # assert input.shape[-1] == self.size
        if self.diffaug:
            input = self.diff_aug_img(input)

        size = input.shape[-1]
        log_size = int(math.log(size, 2))

        cur_size_out = self.conv_in[f'{2 ** log_size}'](input)
        cur_size_out = self.convs[f'{2 ** log_size}'](cur_size_out)

        if alpha < 1:
            down_input = F.interpolate(
                input, scale_factor=0.5, mode='bilinear')
            down_size_out = self.conv_in[f'{2 ** (log_size - 1)}'](down_input)

            out = alpha * cur_size_out + (1 - alpha) * down_size_out
        else:
            out = cur_size_out

        for i in range(log_size - 1, 2, -1):
            out = self.convs[f'{2 ** i}'](out)

        batch, channel, height, width = out.shape

        if self.stddev_group > 0:
            group = min(batch, self.stddev_group)
            # (4, 2, 1, 512//1, 4, 4)
            stddev = out.view(group, -1, self.stddev_feat,
                              channel // self.stddev_feat, height, width)
            # (2, 1, 512//1, 4, 4)
            stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            # (2, 1, 1, 1)
            stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
            # (8, 1, 4, 4)
            stddev = stddev.repeat(group, 1, height, width)
            # (8, 513, 4, 4)
            out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(batch, -1)

        out = self.space_linear(out)

        if summary_ddict is not None:
            with torch.no_grad():
                logits_norm = out.norm(dim=1).mean().item()
                w_norm = self.out_linear.weight.norm(dim=1).mean().item()
                summary_ddict['logits_norm']['logits_norm'] = logits_norm
                summary_ddict['w_norm']['w_norm'] = w_norm

        out = self.out_linear(out)

        latent, position = None, None
        return out, latent, position


class NeRFDiscriminator(MultiScaleDiscriminator):

    def __init__(self, channel_multiplier=2, *args, **kwargs):

        channels = {
            4: 128 * channel_multiplier,
            8: 128 * channel_multiplier,
            16: 128 * channel_multiplier,
            32: 128 * channel_multiplier,
            64: 128 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        MultiScaleDiscriminator.__init__(channels=channels, *args, **kwargs)


class CIPSDiscriminator(nn.Module):

    def __init__(self,
                 diffaug,
                 max_size,
                 channel_multiplier=2,
                 first_downsample=False,
                 stddev_group=0,
                 **kwargs):
        super().__init__()

        self.epoch = 0
        self.step = 0

        self.main_disc = MultiScaleDiscriminator(
            diffaug=diffaug,
            max_size=max_size,
            channel_multiplier=channel_multiplier,
            first_downsample=first_downsample,
            stddev_group=stddev_group)

        self.aux_disc = NeRFDiscriminator(
            diffaug=diffaug,
            max_size=max_size,
            channel_multiplier=channel_multiplier,
            first_downsample=True,
            stddev_group=stddev_group)

    def forward(self,
                input,
                use_aux_disc=False,
                summary_ddict=None,
                alpha=1.,
                **kwargs):

        if use_aux_disc:
            b = input.shape[0] // 2
            main_input = input[:b]
            aux_input = input[b:]
            main_out, latent, position = self.main_disc(
                main_input, alpha, summary_ddict=summary_ddict)
            aux_out, _, _ = self.aux_disc(aux_input, alpha)
            out = torch.cat([main_out, aux_out], dim=0)
        else:
            out, latent, position = self.main_disc(
                input, alpha, summary_ddict=summary_ddict)

        return out, latent, position
