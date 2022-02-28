# Copyright (c) OpenMMLab. All rights reserved.
# style mapping for Generator, seems different form style mapping layers in `inr_nerf_base.py` . # noqa
import torch
import torch.nn as nn

from .module import kaiming_leaky_init


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        assert input.dim() == 2
        return input * torch.rsqrt(
            torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class MultiHeadMappingNetwork(nn.Module):

    def __init__(self,
                 z_dim,
                 hidden_dim,
                 base_layers,
                 head_layers,
                 head_dim_dict,
                 add_norm=False,
                 norm_out=False,
                 **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.head_dim_dict = head_dim_dict

        out_dim = z_dim

        self.module_name_list = []

        self.norm = PixelNorm()

        # base net
        base_net = []
        for i in range(base_layers):
            in_dim = out_dim
            out_dim = hidden_dim

            base_layer_ = nn.Linear(in_features=in_dim, out_features=out_dim)
            base_layer_.apply(kaiming_leaky_init)
            base_net.append(base_layer_)

            if head_layers > 0 or i != base_layers - 1:
                if add_norm:
                    norm_layer_ = nn.LayerNorm(out_dim)
                    base_net.append(norm_layer_)
                act_layer_ = nn.LeakyReLU(0.2, inplace=True)
                base_net.append(act_layer_)

        if len(base_net) > 0:
            if norm_out and head_layers <= 0:
                norm_layer_ = nn.LayerNorm(out_dim)
                base_net.append(norm_layer_)
            self.base_net = nn.Sequential(*base_net)
            self.num_z = 1
            self.module_name_list.append('base_net')
        else:
            self.base_net = None
            self.num_z = len(head_dim_dict)

        # head nets
        head_in_dim = out_dim
        for name, head_dim in head_dim_dict.items():
            if head_layers > 0:
                head_net = []
                out_dim = head_in_dim
                for i in range(head_layers):
                    in_dim = out_dim
                    if i == head_layers - 1:
                        out_dim = head_dim
                    else:
                        out_dim = hidden_dim

                    head_layer_ = nn.Linear(
                        in_features=in_dim, out_features=out_dim)
                    head_layer_.apply(kaiming_leaky_init)
                    head_net.append(head_layer_)

                    if i != head_layers - 1:
                        act_layer_ = nn.LeakyReLU(0.2, inplace=True)
                        head_net.append(act_layer_)
                    else:
                        if norm_out:
                            norm_layer_ = nn.LayerNorm(out_dim)
                            head_net.append(norm_layer_)
                head_net = nn.Sequential(*head_net)
                self.module_name_list.append(name)
            else:
                head_net = nn.Identity()
            self.add_module(name, head_net)

    def forward(self, z):
        """

        :param z:
        :return:
        """

        if self.base_net is not None:
            z = self.norm(z)
            base_fea = self.base_net(z)
            head_inputs = {
                name: base_fea
                for name in self.head_dim_dict.keys()
            }
        else:
            head_inputs = {}
            for idx, name in enumerate(self.head_dim_dict.keys()):
                head_inputs[name] = self.norm(z[idx])

        out_dict = {}
        for name, head_dim in self.head_dim_dict.items():
            head_net = getattr(self, name)
            head_input_ = head_inputs[name]
            out = head_net(head_input_)
            out_dict[name] = out

        return out_dict
