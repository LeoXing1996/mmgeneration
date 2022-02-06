from copy import deepcopy

import torch
import torch.nn as nn

from mmgen.models.architectures.nerf.modules import NoisyReLU


class TestNoisyReLU():

    @classmethod
    def setup_class(cls):
        cls.default_config = dict(
            noise_std=1, inplace=True, end_iteration=4000)

    def test_noisy_relu(self):
        act_fn = NoisyReLU(**self.default_config)
        assert isinstance(act_fn.act, nn.ReLU)
        assert act_fn.act.inplace
        assert act_fn._init_noise_std == 1
        assert act_fn.noise_std == 1
        assert act_fn.training

        inp = torch.randn(2, 3, 3, 3)
        inp_relu = torch.relu(inp)

        # check if noise is added
        out = act_fn(inp.clone())
        assert (out != inp_relu).any()

        # check eval mode
        act_fn.eval()
        out = act_fn(inp.clone())
        assert (out == inp_relu).all()

        # check decrease noise
        act_fn.update_noise(1000)
        assert act_fn.noise_std == 1 - 1 / 4

        act_fn.update_noise(5000)
        assert act_fn.noise_std == 0

        act_fn.update_noise(0)
        assert act_fn.noise_std == 1

        config = deepcopy(self.default_config)
        config['end_iteration'] = 0
        act_fn = NoisyReLU(**config)
        assert act_fn.noise_std == 0
        act_fn.update_noise(10)
        assert act_fn.noise_std == 0
