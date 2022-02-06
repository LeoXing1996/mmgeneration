# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractclassmethod
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn

from mmgen.models import MODELS
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import build_module
from .util import inverse_transform_sampling


@MODELS.register_module()
class BaseNeRF(nn.Module, metaclass=ABCMeta):
    """BaseNeRF Module.

    Args:
        camera (dict): config for camera.
    """

    def __init__(self,
                 camera,
                 hierarchical_sampling=None,
                 n_importance=None,
                 num_samples_per_ray=64,
                 white_background=False):
        # do not use ``super().__init__()`` here, avoid error in imultiple
        # inheritance
        nn.Module.__init__(self)

        self._camera_cfg = deepcopy(camera)
        # set default type
        if 'type' not in self._camera_cfg:
            self._camera_cfg['type'] = 'PoseCamera'
        self.camera = build_module(self._camera_cfg)

        self.n_importance = n_importance
        self.n_samples = num_samples_per_ray
        self.white_background = white_background
        self.hierarchical_sampling = hierarchical_sampling

        # buffer to save images in evaluation
        self._eval_img_buffer = dict()
        self._eval_buffer_iter = -1

    def eval(self):
        """Overwrite module.eval, in order to set eval mode for ray sampler."""
        super().eval()
        self.camera.ray_sampler.eval()

    def train(self, mode=True):
        """Overwrite module.train, in order to set train mode for ray sampler.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                mode (``False``). Defaults to ``True``.
        """
        super().train(mode)
        self.camera.ray_sampler.train(mode=mode)

    @property
    def eval_img_buffer(self):
        return self._eval_img_buffer

    @property
    def eval_img_buffer_iter(self):
        return self._eval_buffer_iter

    def collect_img_buffer(self):
        """Collect list to tensor, this function can be called by vis hooks and
        eval hooks.

        This operation is dangerous, maybe we can find a better solution.
        """
        for k, v in self._eval_img_buffer.items():
            if isinstance(v, list):
                self._eval_img_buffer[k] = torch.cat(v, dim=0)
        torch.cuda.empty_cache()

    def _update_img_buffer(self, output_dict):
        """Function for evaluation.

        Different from `self.train_step`, we save the output_dict to the image
        buffer. Then EvalHook and VisHook can load images from this buffer.
        """
        # not found self.iteration, called byhh evaluation.py Do nothing
        if not hasattr(self, 'iteration'):
            return
        if self._eval_img_buffer:
            assert all([k in output_dict for k in self._eval_img_buffer])
        for k, v in output_dict.items():
            if self.eval_img_buffer_iter == self.iteration:
                # the same iteration, add new element
                self._eval_img_buffer[k].append(v.cpu())
            else:
                # not the same iteration, assign new elements
                self._eval_img_buffer[k] = [v.cpu()]
        self._eval_buffer_iter = self.iteration

    def _get_nerf_loss(self, outputs_dict):
        losses_dict = {}
        for loss_module in self.nerf_loss:
            loss_ = loss_module(outputs_dict)
            if loss_ is None:
                continue
            losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self._parse_losses(losses_dict)
        return loss, log_var

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            # Allow setting None for some loss item.
            # This is to support dynamic loss module, where the loss is
            # calculated with a fixed frequency.
            elif loss_value is None:
                continue
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        # Note that you have to add 'loss' in name of the items that will be
        # included in back propagation.
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def gaussian_noise(self, tar_shape):
        return torch.randn(tar_shape).to(get_module_device(self))

    def uniform_noise(self, tar_shape):
        return torch.rand(tar_shape).to(get_module_device(self))

    def _add_noise_to_z(self, z_vals, noise_fn):
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)

        t_rand = noise_fn(list(z_vals.shape))
        z_vals = lower + (upper - lower) * t_rand
        return z_vals

    def get_weight(self, z_vals, ray_vectors, sigma):
        """Calculate weight in Eq TODO
        Args:
            z_vals: [n_points * n_samples]
            ray_vectors: [n_points, 4]
        Returns:
            torch.Tensor: size as [n_points, n_samples]
        """
        deltas = z_vals[..., 1:] - z_vals[..., :-1]
        delta_inf = torch.ones_like(deltas[..., :1]) * 1e10
        deltas = torch.cat([deltas, delta_inf], dim=-1)

        # [n_points, n_sample] * [n_points, 1]
        deltas = deltas * torch.norm(ray_vectors, dim=-1, keepdim=True)
        # alpha = 1. - torch.exp(-F.relu(sigma) * deltas)
        alpha = 1. - torch.exp(-sigma * deltas)

        ones = torch.ones_like(alpha[..., :1])
        weights = alpha * torch.cumprod(
            torch.cat([ones, 1. - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
        return weights

    def volume_rendering(self, raw_output, z_vals, ray_vectors, is_fine=False):
        # TODO: support noise in volume rendering

        # NOTE: we should consist the name here
        rgbs, sigmas = raw_output['rgbs'], raw_output['alphas']

        # [n_points, n_samples, 1] to [n_points, n_samples]
        z_vals, sigmas = z_vals.squeeze(), sigmas.squeeze()

        # [n_points, n_samples]
        weights = self.get_weight(z_vals, ray_vectors, sigmas)

        # [bz, n_points, n_samples, 1] * [bz, n_points, n_samples, 3]
        # rgb_final = torch.sum(weights[..., None] * torch.sigmoid(rgbs), -2)
        rgb_map = torch.sum(weights[..., None] * rgbs, -2)
        # [bz, n_points, n_samples] * [bz, n_points, n_samples, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disparity map, which is inversely proportional to the depth
        # [bz, n_points]
        # TODO: here we should use more method to avoid nan than official code
        # in the first around of the training, weights = 0 makes
        # disp_final to nan
        disp_map = 1. / torch.max(depth_map / torch.sum(weights, -1),
                                  1e-10 * torch.ones_like(depth_map))
        # [bz, n_points]
        acc_map = torch.sum(weights, -1)

        if self.white_background:
            # NOTE: add comment here, why we calculate this
            rgb_map = rgb_map + (1. - acc_map[..., None])

        suffix = 'final' if (is_fine or not self.hierarchical_sampling) \
            else 'coarse'
        output_dict = {
            f'rgb_{suffix}': rgb_map,
            f'depth_{suffix}': depth_map,
            f'disp_{suffix}': disp_map,
            f'acc_{suffix}': acc_map,
            f'weights_{suffix}': weights
        }
        return output_dict

    def prepare_hierarchical_sampling(self, z_vals, weights, camera_pose,
                                      rays):

        # z_vals: [1, n_samples] to [n_points, n_samples]
        z_vals = z_vals.expand(weights.shape[0], -1)
        z_vals_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1]).squeeze()
        # z_samples: [n_points, n_samples']
        z_samples = inverse_transform_sampling(
            z_vals_mid,
            weights[..., 1:-1],
            self.n_importance,
            det=not self.training,
            device=get_module_device(self)).detach()

        # z_zals_fine: [n_points, n_samples_fine]
        z_vals_fine, _ = torch.sort(
            torch.cat([z_vals, z_samples], dim=1), dim=1)

        points_fine = self.camera.sample_points(rays, z_vals_fine, camera_pose)
        return points_fine, z_vals_fine

    def forward_network_batchify(self, data_dict, network, **kwargs):
        """Forward the network in chunk. All tensors in ``data_dict`` are shape
        like `[n_points, n_samples, n]`.

        Args:
            data_dict (dict): Dict of data feed to the network.
            network (torch.nn.Module): The network used to forward.

        Returns:
        """
        data_keys = list(data_dict.keys())
        # 0. save input shape
        n_points, n_samples, _ = data_dict[data_keys[0]].shape
        # total pixels to render
        n_pixels = n_points * n_samples
        # 1. flatten the input
        data_dict_flatten = {
            k: data_dict[k].reshape(-1, data_dict[k].shape[-1])
            for k in data_keys
        }
        # shape checking
        assert all([
            data_dict_flatten[k].shape[0] == n_pixels for k in data_keys
        ]), ('n_pixels in renderer inputs not consistent.')

        # 2. init list to save output
        raw_output_list = []
        network_chunk = n_pixels if (
            self.network_chunk is None
            or self.network_chunk == -1) else self.network_chunk

        # 3. loop for each mini-batch
        for idx_s in range(0, n_pixels, network_chunk):
            idx_e = idx_s + network_chunk
            mini_data_dict = {
                k: data_dict_flatten[k][idx_s:idx_e]
                for k in data_keys
            }
            raw_output_ = network(**mini_data_dict, **kwargs)
            raw_output_list.append(raw_output_)

        # 4 gather the output dict
        raw_output = self.concatenate_dict(raw_output_list)

        # 5. reshape to [n_points (render_chunk), n_samples, -1]
        for k in raw_output.keys():
            raw_output[k] = raw_output[k].reshape([n_points, n_samples, -1])
        return raw_output

    def update_noise(self, model, curr_iter):
        """Update noise used in model.

        Args:
            model (nn.Module): Model to decreasing noise.
            curr_iter (int): Current iteration.
        """
        for _, m in model.named_modules():
            if hasattr(m, 'update_noise'):
                m.update_noise(curr_iter)

    @staticmethod
    def concatenate_dict(list_of_dict):
        """
        Args:
            list_of_dict ([dict, ]): A list contains dicts to be concatenated.
                All dict have the same keys and values are tensors can be
                concatenated at the first dimension.
        """
        new_dict = dict()
        tar_keys = list_of_dict[0].keys()
        for k in tar_keys:
            new_dict[k] = torch.cat([d[k] for d in list_of_dict], dim=0)
        return new_dict

    @abstractclassmethod
    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None,
                   **kwargs):
        """Train step."""

    def forward_test(self, data, **kwargs):
        """Testing function for NeRF models.

        Args:
            data (torch.Tensor | dict | None): Input data. This data will be
                passed to different methods.
        """
        mode = kwargs.pop('mode', 'sampling')
        if mode == 'reconstruction':
            return self.reconstruction_step(data, **kwargs)
        elif mode == 'sampling':
            return self.sample_from_noise(data, **kwargs)
        else:
            raise NotImplementedError('Other specific testing functions should'
                                      ' be implemented by the sub-classes.')

    def forward(self, data, return_loss=False, **kwargs):
        """Forward function.

        Args:
            data (dict | torch.Tensor): Input data dictionary.
            return_loss (bool, optional): Whether in training or testing.
                Defaults to False.

        Returns:
            dict: Output dictionary.
        """
        if return_loss:
            return self.forward_train(data, **kwargs)

        return self.forward_test(data, **kwargs)
