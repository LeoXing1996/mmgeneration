# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import ACTIVATION_LAYERS
from numpy import pi

from mmgen.models.builder import MODULES


@MODULES.register_module()
class NeRFPositionalEmbedding(nn.Module):
    """Base embedder for nerf-models.

    Args:
        ignore_pi: refers to https://github.com/bmild/nerf/issues/12
    """

    def __init__(self,
                 n_freq,
                 embedding_method='sin',
                 include_input=False,
                 ignore_pi=False):
        super().__init__()
        self.length = n_freq
        self.embedding_method = embedding_method
        self.ignore_pi = ignore_pi
        self.emb_fn = partial(
            getattr(self, f'{embedding_method}_embedding'),
            n_freq=n_freq,
            ignore_pi=ignore_pi)
        self.include_input = include_input

    @property
    def embedding_factor(self):
        if self.embedding_method == 'sin':
            dim = self.length * 2 + 1 if self.include_input \
                else self.length * 2
        return dim

    @staticmethod
    def sin_embedding(x, n_freq, ignore_pi):
        factor = 1 if ignore_pi else pi
        emb_res = torch.cat([
            torch.cat([
                torch.sin((2**i) * factor * x),
                torch.cos((2**i) * factor * x)
            ],
                      dim=-1) for i in range(n_freq)
        ],
                            dim=-1)
        return emb_res

    @staticmethod
    def gaussian_embedding(x, n_freq):
        """Embedding method used in griaffe. TODO
        Args:
            x (torch.Tensor)
            n_freq ()

        Returns:
            torch.Tensor: Embedded position.
        """
        pass

    def forward(self, x):
        """"""
        embedding = self.emb_fn(x)
        if self.include_input:
            embedding = torch.cat([x, embedding], dim=-1)
        return embedding

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(method={self.embedding_method}, L_freq={self.length}, '
                     f'include_input={self.include_input}, '
                     f'ignore_pi={self.ignore_pi}, '
                     f'embedding_factor={self.embedding_factor})')
        return repr_str


@ACTIVATION_LAYERS.register_module()
class NoisyReLU(nn.Module):
    r"""ReLU activation function with gaussian noise. In training mode,

    .. math::

    NoisyReLU(x) = ReLU(x) + z, z \sim N(0, raw_std)

    And in evaluation mode, noise is not used and perform same as standard ReLU
    activation.

    And standard deviation of the noise can be modified by
    ``self.update_noise()``. (This operation cannot be done by hooks
    because we may want to change noise scale between discriminator updating
    and generation updating.)

    Args:
        noise_std (float | int, optional): The initial standard deviation of
            the noise, must be larger than or equals to `0`. Defaults to `1`.
        inplace (bool, optional): Whether use inplace version of ReLU. Defaults
            to ``False``.
        end_iteration (int, optional): Then end iteration of the noise
            decreasing operation. If less than or equal to 0, noise decreasing
            will not be performed. Defaults to `-1`.
    """

    def __init__(self, noise_std=1, inplace=False, end_iteration=-1):
        super().__init__()
        self.act = nn.ReLU(inplace=inplace)

        assert isinstance(
            noise_std,
            (int, float)), ('\'noise_std\' must be int or float, but receive '
                            f'\'{type(noise_std)}\'.')
        assert noise_std >= 0, (
            '\'noise_std\' must be larger than or equals to \'0\', but '
            f'receive \'{noise_std}\'.')
        self._init_noise_std = noise_std
        self.end_iter = end_iteration
        self.noise_std = noise_std if end_iteration > 0 else 0

    def update_noise(self, curr_iter):
        """Update the scale of noise.

        Args:
            curr_iter (int): Current iteration.
        """
        # do not decreasing (end_iter < 0 <= curr_iter) or has finished
        if self.end_iter <= curr_iter:
            self.noise_std = 0
        else:
            self.noise_std = self._init_noise_std - \
                self._init_noise_std / self.end_iter * curr_iter

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor after activation.
        """
        if self.training:
            return self.act(x + torch.randn_like(x) * self.noise_std)
        return self.act(x)


def flatten_and_clip_input(input):
    # clip homogeneuous coordinates to non-homogeneuous ones
    input_ = input[..., :3]
    input_flatten = input_.reshape([-1, 3])
    return input_flatten
