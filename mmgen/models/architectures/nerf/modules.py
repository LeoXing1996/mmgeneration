# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import torch
import torch.nn as nn
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


def flatten_and_clip_input(input):
    # clip homogeneuous coordinates to non-homogeneuous ones
    input_ = input[..., :3]
    input_flatten = input_.reshape([-1, 3])
    return input_flatten
