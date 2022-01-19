from abc import ABCMeta, abstractmethod

import numpy as np
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class NeRFRaySampleHook(Hook, metaclass=ABCMeta):

    def __init__(self, interval=1):
        """Update ray sample strategy before each train iteration.

        Args:
            interval (int): Interval to update ray sampler strategy. Defaults
                to 1.
        """
        super().__init__()
        self.interval = interval

    def before_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model
        ray_sampler = model.camera.ray_sampler
        curr_iter = runner.iter

        # update ray sampler
        self._update_ray_sampler(ray_sampler, curr_iter)

    @abstractmethod
    def _update_ray_sampler(self, ray_sampler, curr_iter):
        """Update the ray sampler.

        Args:
            ray_sampler (object): The ray sampler to be updated.
            curr_iter (int): Current iteration.
        """


@HOOKS.register_module()
class PrecropRaySamplerHook(NeRFRaySampleHook):

    def __init__(self, precrop_frac_iter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precrop_frac_iter = precrop_frac_iter

    def _update_ray_sampler(self, ray_sampler, curr_iter):
        """Update ray sampler.

        Args:
            ray_sampler (object)
            curr_iter (int)
        """
        precrop_frac = ray_sampler.precrop_frac \
            if curr_iter < self.precrop_frac_iter else 1
        ray_sampler.set_precrop_frac(precrop_frac)


@HOOKS.register_module()
class FlexGridRaySamplerHook(NeRFRaySampleHook):

    def __init__(self, scale_annel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_annel = scale_annel

    def _update_ray_sampler(self, ray_sampler, curr_iter):
        min_scale = ray_sampler.min_scale
        max_scale = ray_sampler.max_scale

        if self.scale_annel > 0:
            k_iter = curr_iter // 1000 * 3

            curr_min_scale = max(
                min_scale, max_scale * np.exp(-k_iter * self.scale_annel))
            curr_min_scale = min(0.9, curr_min_scale)
        else:
            curr_min_scale = min_scale
        ray_sampler.set_min_scale(curr_min_scale)
