# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmgen.models.builder import MODULES
from .diffaug import DiffAugment
from .module import ConvLayer, EqualLinear, ResBlock


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
        MultiScaleDiscriminator.__init__(
            self, channels=channels, *args, **kwargs)


@MODULES.register_module()
class CIPS3DDiscriminator(nn.Module):

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

    def forward(
            self,
            input,
            use_aux_disc=False,
            summary_ddict=None,
            alpha=1.,
            return_latent=False,  # add by us
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

        if return_latent:
            return out, latent, position
        return out
