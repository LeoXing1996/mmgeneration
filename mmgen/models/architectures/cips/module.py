# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.ops.fused_bias_leakyrelu import (FusedBiasLeakyReLU,
                                           fused_bias_leakyrelu)
from mmcv.ops.upfirdn2d import upfirdn2d


def frequency_init(freq):

    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq,
                                  np.sqrt(6 / num_input) / freq)

    return init


def kaiming_leaky_init(m):
    """Init the mapping network of StyleGAN. fc -> leaky_relu -> fc -> ... Note
    the outputs of each fc, especially when the number of layers increases.

    :param m:
    :return:
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(
            m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class SkipLayer(nn.Module):

    def __init__(self):
        super(SkipLayer, self).__init__()

    def forward(self, x0, x1):
        # out = (x0 + x1) / math.pi
        out = x0 + x1
        return out


class ScaledLeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class SinAct(nn.Module):

    def __init__(self):
        super(SinAct, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class SinStyleMod(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=1,
                 style_dim=None,
                 use_style_fc=False,
                 demodulate=True,
                 use_group_conv=False,
                 eps=1e-8,
                 **kwargs):
        """

        :param in_channel:
        :param out_channel:
        :param kernel_size:
        :param style_dim: =in_channel
        :param use_style_fc:
        :param demodulate:
        """
        super().__init__()

        self.eps = eps
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.style_dim = style_dim
        self.use_style_fc = use_style_fc
        self.demodulate = demodulate
        self.use_group_conv = use_group_conv

        self.padding = kernel_size // 2

        if use_group_conv:
            self.weight = nn.Parameter(
                torch.randn(1, out_channel, in_channel, kernel_size,
                            kernel_size))
            torch.nn.init.kaiming_normal_(
                self.weight[0],
                a=0.2,
                mode='fan_in',
                nonlinearity='leaky_relu')
        else:
            assert kernel_size == 1
            self.weight = nn.Parameter(torch.randn(1, in_channel, out_channel))
            torch.nn.init.kaiming_normal_(
                self.weight[0],
                a=0.2,
                mode='fan_in',
                nonlinearity='leaky_relu')

        if use_style_fc:
            self.modulation = nn.Linear(style_dim, in_channel)
            self.modulation.apply(kaiming_leaky_init)
        else:
            self.style_dim = in_channel

        self.sin = SinAct()
        self.norm = nn.LayerNorm(in_channel)

    def forward_bmm(self, x, style, weight):
        """

        :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
        :param style: (b, in_c)
        :return:
        """
        assert x.shape[0] == style.shape[0]
        if x.dim() == 2:
            input = rearrange(x, 'b c -> b 1 c')
        elif x.dim() == 3:
            input = x
        else:
            assert 0

        batch, N, in_channel = input.shape

        if self.use_style_fc:
            # style = self.sin(style)
            style = self.modulation(style)
            # style = self.norm(style)
            style = style.view(-1, in_channel, 1)
        else:
            # style = self.norm(style)
            style = rearrange(style, 'b c -> b c 1')
            # style = style + 1.

        # (1, in, out) * (b in 1) -> (b, in, out)
        weight = weight * (style + 1)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([
                1,
            ]) + self.eps)  # (b, out)
            # (b, in, out) * (b, 1, out) -> (b, in, out)
            weight = weight * demod.view(batch, 1, self.out_channel)
        # (b, n, in) * (b, in, out) -> (b, n, out)
        out = torch.bmm(input, weight)

        if x.dim() == 2:
            out = rearrange(out, 'b 1 c -> b c')
        elif x.dim() == 3:
            # out = rearrange(out, "b n c -> b n c")
            pass
        return out

    def forward_group_conv(self, x, style):
        """

        :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
        :param style: (b, in_c)
        :return:
        """
        assert x.shape[0] == style.shape[0]
        if x.dim() == 2:
            input = rearrange(x, 'b c -> b c 1 1')
        elif x.dim() == 3:
            input = rearrange(x, 'b n c -> b c n 1')
        elif x.dim() == 4:
            input = x
        else:
            assert 0

        batch, in_channel, height, width = input.shape

        if self.use_style_fc:
            style = self.modulation(style).view(-1, 1, in_channel, 1, 1)
            style = style + 1.
        else:
            style = rearrange(style, 'b c -> b 1 c 1 1')
            # style = style + 1.
        # (1, out, in, ks, ks) * (b, 1, in, 1, 1) -> (b, out, in, ks, ks)
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) +
                                self.eps)  # (b, out)
            # (b, out, in, ks, ks) * (b, out, 1, 1, 1)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        # (b*out, in, ks, ks)
        weight = weight.view(batch * self.out_channel, in_channel,
                             self.kernel_size, self.kernel_size)
        # (1, b*in, h, w)
        input = input.reshape(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        if x.dim() == 2:
            out = rearrange(out, 'b c 1 1 -> b c')
        elif x.dim() == 3:
            out = rearrange(out, 'b c n 1 -> b n c')

        return out

    def forward(self, x, style, force_bmm=False):
        """

        :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
        :param style: (b, in_c)
        :return:
        """
        if self.use_group_conv:
            if force_bmm:
                weight = rearrange(self.weight, '1 out in 1 1 -> 1 in out')
                out = self.forward_bmm(x=x, style=style, weight=weight)
            else:
                out = self.forward_group_conv(x=x, style=style)
        else:
            out = self.forward_bmm(x=x, style=style, weight=self.weight)
        return out


class EqualLinear(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 bias_init=0,
                 lr_mul=1.,
                 scale=None,
                 norm_weight=False,
                 activate=None,
                 **kwargs):
        """

        :param in_dim:
        :param out_dim:
        :param bias:
        :param bias_init:
        :param lr_mul: 0.01
        """
        super().__init__()

        self.lr_mul = lr_mul
        self.norm_weight = norm_weight

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        if scale is not None:
            self.scale = scale
        else:
            self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.activate = activate

    def forward(self, input):
        """

        :param input: (b c), (b, n, c)
        :return:
        """
        if self.norm_weight:
            demod = torch.rsqrt(
                self.weight.pow(2).sum([
                    1,
                ], keepdim=True) + 1e-8)
            weight = self.weight * demod
        else:
            weight = self.weight
        if self.activate:
            out = F.linear(input, weight * self.scale)
            out = fused_bias_leakyrelu(out)
        else:
            out = F.linear(
                input, weight * self.scale, bias=self.bias * self.lr_mul)
        return out


class EqualConv2d(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out


class EqualConvTranspose2d(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out


class ToRGB(nn.Module):

    def __init__(self, in_dim, dim_rgb=3, use_equal_fc=False):
        super().__init__()
        self.in_dim = in_dim
        self.dim_rgb = dim_rgb

        if use_equal_fc:
            self.linear = EqualLinear(in_dim, dim_rgb, scale=1.0)
        else:
            self.linear = nn.Linear(in_dim, dim_rgb)
        pass

    def forward(self, input, skip=None):

        out = self.linear(input)

        if skip is not None:
            out = out + skip
        return out


class SinBlock(nn.Module):

    def __init__(self, in_dim, out_dim, style_dim, name_prefix):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.name_prefix = name_prefix

        self.style_dim_dict = {}

        self.mod1 = SinStyleMod(
            in_channel=in_dim,
            out_channel=out_dim,
            style_dim=style_dim,
            use_style_fc=True)
        self.style_dim_dict[f'{name_prefix}_0'] = self.mod1.style_dim
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.mod2 = SinStyleMod(
            in_channel=out_dim,
            out_channel=out_dim,
            style_dim=style_dim,
            use_style_fc=True)
        self.style_dim_dict[f'{name_prefix}_1'] = self.mod2.style_dim
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.skip = SkipLayer()

    def forward(self, x, style_dict, skip=False):
        x_orig = x

        style = style_dict[f'{self.name_prefix}_0']
        x = self.mod1(x, style)
        x = self.act1(x)

        style = style_dict[f'{self.name_prefix}_1']
        x = self.mod2(x, style)
        out = self.act2(x)

        if skip and out.shape[-1] == x_orig.shape[-1]:
            out = self.skip(out, x_orig)
        return out


class LinearBlock(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        name_prefix,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.name_prefix = name_prefix

        self.net = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=out_dim, out_features=out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net.apply(kaiming_leaky_init)

    def forward(self, x, *args, **kwargs):
        out = self.net(x)
        return out


class ConvLayer(nn.Sequential):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 bias=True,
                 activate=True,
                 upsample=False,
                 padding='zero'):

        layers = OrderedDict()

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            # layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            layers['down_blur'] = Blur(blur_kernel, pad=(pad0, pad1))

            stride = 2

        if upsample:
            up_conv = EqualConvTranspose2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=0,
                stride=2,
                bias=bias and not activate,
            )
            # layers.append(up_conv)
            layers['up_conv'] = up_conv

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            # layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
            layers['up_blur'] = Blur(blur_kernel, pad=(pad0, pad1))

        else:
            if not downsample:
                if padding == 'zero':
                    self.padding = (kernel_size - 1) // 2

                elif padding == 'reflect':
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        # layers.append(nn.ReflectionPad2d(padding))
                        layers['pad'] = nn.ReflectionPad2d(padding)

                    self.padding = 0

                elif padding != 'valid':
                    raise ValueError(
                        'Padding should be "zero", "reflect", or "valid"')

            equal_conv = EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
            # layers.append(equal_conv)
            layers['equal_conv'] = equal_conv

        if activate:
            if bias:
                layers['flrelu'] = FusedBiasLeakyReLU(out_channel)

            else:
                layers['slrelu'] = ScaledLeakyReLU(0.2)

        super().__init__(layers)


class ResBlock(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 blur_kernel=[1, 3, 3, 1],
                 kernel_size=3,
                 downsample=True,
                 first_downsample=False):
        super().__init__()

        if first_downsample:
            self.conv1 = ConvLayer(
                in_channel, in_channel, kernel_size, downsample=downsample)
            self.conv2 = ConvLayer(in_channel, out_channel, kernel_size)
        else:
            self.conv1 = ConvLayer(in_channel, in_channel, kernel_size)
            self.conv2 = ConvLayer(
                in_channel, out_channel, kernel_size, downsample=downsample)

        self.skip = ConvLayer(
            in_channel,
            out_channel,
            1,
            downsample=downsample,
            activate=False,
            bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


# --> blocks for NeRF


class UniformBoxWarp(nn.Module):

    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2 / sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor


class LinearScale(nn.Module):

    def __init__(self, scale, bias):
        super(LinearScale, self).__init__()
        self.scale_v = scale
        self.bias_v = bias
        pass

    def forward(self, x):
        out = x * self.scale_v + self.bias_v
        return out


class FiLMLayer(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 style_dim,
                 use_style_fc=True,
                 which_linear=nn.Linear,
                 **kwargs):
        super(FiLMLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.use_style_fc = use_style_fc

        self.linear = which_linear(in_dim, out_dim)
        self.linear.apply(frequency_init(25))

        self.gain_scale = LinearScale(scale=15, bias=30)
        # Prepare gain and bias layers
        if use_style_fc:
            self.gain_fc = which_linear(style_dim, out_dim)
            self.bias_fc = which_linear(style_dim, out_dim)
            self.gain_fc.weight.data.mul_(0.25)
            self.bias_fc.weight.data.mul_(0.25)
        else:
            self.style_dim = out_dim * 2

    def forward(self, x, style):
        """

        :param x: (b, c) or (b, n, c)
        :param style: (b, c)
        :return:
        """

        if self.use_style_fc:
            gain = self.gain_fc(style)
            gain = self.gain_scale(gain)
            bias = self.bias_fc(style)
        else:
            style = rearrange(style, 'b (n c) -> b n c', n=2)
            gain, bias = style.unbind(dim=1)
            gain = self.gain_scale(gain)

        if x.dim() == 3:
            gain = rearrange(gain, 'b c -> b 1 c')
            bias = rearrange(bias, 'b c -> b 1 c')
        elif x.dim() == 2:
            pass
        else:
            assert 0

        x = self.linear(x)
        out = torch.sin(gain * x + bias)
        return out
