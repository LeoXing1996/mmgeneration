import torch.nn as nn
from mmcv.cnn.bricks.activation import build_activation_layer

from mmgen.models.architectures.biggan.modules import SNConvModule


class GRAFDiscBlock(nn.Module):

    def __init__(self,
                 in_chn,
                 out_chn,
                 kernel_size,
                 stride,
                 padding,
                 act_cfg,
                 use_IN=False):
        super().__init__()
        blocks = [
            SNConvModule(
                in_channels=in_chn,
                out_channels=out_chn,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                act_cfg=None,
                with_spectral_norm=True,
                bias=False)
        ]
        if use_IN:
            blocks.append(nn.InstanceNorm2d(out_chn))
        if act_cfg:
            blocks.append(build_activation_layer(act_cfg))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
