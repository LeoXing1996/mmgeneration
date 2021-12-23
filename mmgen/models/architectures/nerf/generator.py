# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks import build_activation_layer
from mmcv.runner import load_checkpoint

from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger
from .modules import NeRFPositionalEmbedding, flatten_and_clip_input


@MODULES.register_module()
class NeRFRenderer(nn.Module):
    """Generator of NeRF.

    Args:
        num_layers (int)
        base_channels (int):
        input_ch (int): Input channels, defaults to 3.
        input_ch_views (int): Input channels of view ?
        skips (list[int]):
        use_viewdirs (bool): Whether use camera view and directions
    """

    def __init__(self,
                 num_layers=8,
                 base_channels=256,
                 input_ch=3,
                 input_ch_views=3,
                 skips=[4],
                 alpha_act_cfg=dict(type='ReLU'),
                 rgb_act_cfg=dict(type='Sigmoid'),
                 use_viewdirs=True,
                 pose_embedding=None,
                 points_embedding=None):
        super().__init__()
        self.D = num_layers
        self.W = base_channels
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # init embedding and update input channels
        if pose_embedding and self.use_viewdirs:
            self.pose_embedding = NeRFPositionalEmbedding(**pose_embedding)
            input_ch_views *= self.pose_embedding.embedding_factor
        else:
            self.pose_embedding = None
        if points_embedding:
            self.points_embedding = NeRFPositionalEmbedding(**points_embedding)
            input_ch *= self.points_embedding.embedding_factor
        else:
            self.points_embedding = None

        self.alpha_act = build_activation_layer(alpha_act_cfg)
        self.rgb_act = build_activation_layer(rgb_act_cfg)

        pts_linears = [nn.Linear(input_ch, base_channels)]
        for idx in range(num_layers - 1):
            in_channels = base_channels + input_ch \
                if idx in self.skips else base_channels
            pts_linears += [nn.Linear(in_channels, base_channels)]
        self.pts_linears = nn.ModuleList(pts_linears)

        # use camera view and directions
        if use_viewdirs:
            # Implementation according to the official code release
            # https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105
            self.views_linears = nn.ModuleList([
                nn.Linear(input_ch_views + base_channels, base_channels // 2)
            ])

            self.feature_linear = nn.Linear(base_channels, base_channels)
            self.alpha_linear = nn.Linear(base_channels, 1)
            self.rgb_linear = nn.Linear(base_channels // 2, 3)
        else:
            self.output_linear = nn.Linear(base_channels, 4)

        self.init_weights(pretrained=None, strict=True)

    def forward(self, points, views=None):
        """ Forward function.
        Args:
            points (torch.Tensor): Shape as [n_points', n_samples, 4]
            views (torch.Tensor): Shape as [n_points', n_samples, 4]
            chunk: batch chunk
        n_points' for mini-n-points

        Returns:
            torch.Tensor: [rgb, alpha].
        """
        if self.use_viewdirs:
            views = flatten_and_clip_input(views)
            if self.pose_embedding:
                views = self.pose_embedding(views)

        points = flatten_and_clip_input(points)
        if self.points_embedding:
            points = self.points_embedding(points)

        h = points
        for i in range(len(self.pts_linears)):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([points, h], -1)

        if self.use_viewdirs:
            assert views is not None, (
                '\'views\' must be passed for \'use_viewdirs=True\'.')
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, views], -1)

            for i in range(len(self.views_linears)):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
        else:
            outputs = self.output_linear(h)
            rgb, alpha = torch.split(outputs, [3, 1], dim=-1)

        rgb = self.rgb_act(rgb)
        alpha = self.alpha_act(alpha)
        output_dict = dict(rgbs=rgb, alphas=alpha)
        return output_dict

    def init_weights(self, pretrained=None, strict=True):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether to allow different params for the
                model and checkpoint. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            # weight: xavier_uniform
            # biase: zero
            for _, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    xavier_init(m, bias=0, distribution='uniform')
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
