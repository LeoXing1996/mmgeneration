import numpy as np
import torch

from ..builder import MODULES


@MODULES.register_module()
class UniformTimeStepSampler():

    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps

    def sample(self, batch_size=0):
        # return torch.randint(0, self.num_timesteps, (batch_size,))
        p = [1 / self.num_timesteps for _ in range(self.num_timesteps)]
        return torch.from_numpy(
            np.random.choice(self.num_timesteps, size=(batch_size, ), p=p))

    def __call__(self, batch_size):
        return self.sample(batch_size)


@MODULES.register_module()
class LossWeightSampler:
    """[WIP] Timestep sampler based on training weights."""

    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weight = None

    def sample(self):
        pass

    def weight(self):
        return self._weight

    def update_weight(self):
        pass

    def __call__(self, batch_size):
        return self.sample(batch_size)
