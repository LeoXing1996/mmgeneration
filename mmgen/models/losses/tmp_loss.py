from copy import deepcopy
from functools import partial

import mmcv
import torch
import torch.distributed as dist
import torch.nn as nn

from mmgen.models.builder import MODULES
from .pixelwise_loss import mse_loss
from .tmp_pix_loss import (DiscretizedGaussianLogLikelihoodLoss,
                           GaussianKLDLoss, _reduction_modes)
from .utils import reduce_loss


class DDPMLoss(nn.Module):
    """Base module for DDPM losses. We support loss rescale and log collection
    for DDPM models in this module.

    We support two kinds of loss rescale methods, which can be
    controlled by ``rescale_mode`` and ``rescale_cfg``:
    1. ``rescale_mode == 'constant'``: ``constant_rescale`` would be called,
        and ``rescale_cfg`` should be passed as ``dict(scale=SCALE)``. Then,
        all loss terms would be rescaled by multiply with ``SCALE``
    2. ``rescale_mode == dynamic_weight``: ``dynamic_weight_rescale`` would be
        called, and ``weight`` or ``sampler`` who contains attribute of weight
        must be passed. Then, loss of timestep `t` would be multiplied with
        `weight[t]`. ``scale`` is allowed as well in ``rescale_cfg``.
        To be noted that, ``weight`` would be inplace modified in the outter
        code.
    If ``rescale_mode`` is not passed, ``rescale_cfg`` would be ignored, and
    all loss terms would not be rescaled.

    For losses log collection, we support users to pass a list of
    (or single) config by ``log_cfgs`` argument to define how they want to
    collect loss terms and show them in the log.
    Each log configs must contain ``type`` keyword, and may contain ``prefix``
    and ``reduction`` keywords.

    ``type``: Use to get the corresponding collection function. Functions would
        be named as ``f'{type}_log_collect'``.
    ``prefix``: Control the prefix of output when print logs. If passed, it
        must start with ``'loss_'``. If not passed, ``'loss_'`` would be used.
    ``reduction``: Control the reduction method of the collected loss terms.

    We implement ``quartile_log_collection`` in this module. Detailly, we
    divide total timesteps into four parts and collect the loss in the
    corresponding timestep intervals.

    To use those collection methods, users may passed ``log_cfgs`` as the
    following example:

    .. code-block:: python
        :linenos:

        log_cfgs = [
            dict(type='quartile', reduction=REUCTION, prefix_name=PREFIX),
            ...
        ]

    Args:
        log_cfgs (list[dict] | dict | optional): Configs to collect logs.
            Defaults to None.
        rescale_mode (str, optional): Mode of the loss rescale method.
            Defaults to `''`.
        rescale_cfg (dict, optional): Config of the loss rescale method.
        sampler (object): Weight sampler. Defaults to None.
        weight (torch.Tensor, optional): Weight used for rescale losses.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        loss_name (str, optional): Name of the loss item. Defaults to `''`.
    """

    def __init__(self,
                 log_cfgs,
                 rescale_mode='',
                 rescale_cfg=None,
                 weight=None,
                 sampler=None,
                 reduction='mean',
                 loss_name=''):
        super().__init__()
        self.log_fn_list = []

        log_cfgs_ = deepcopy(log_cfgs)
        if log_cfgs_ is not None:
            if not isinstance(log_cfgs_, list):
                log_cfgs_ = [log_cfgs_]
            assert mmcv.is_list_of(log_cfgs_, dict)
            for log_cfg_ in log_cfgs_:
                log_type = log_cfg_.pop('type')
                log_collect_fn = f'{log_type}_log_collect'
                assert hasattr(self, log_collect_fn)
                log_collect_fn = getattr(self, log_collect_fn)

                log_cfg_.setdefault('prefix_name', 'loss')
                assert log_cfg_['prefix_name'].startswith('loss')
                log_cfg_.setdefault('reduction', reduction)

                self.log_fn_list.append(partial(log_collect_fn, **log_cfg_))
        self.log_vars = dict()

        self._loss_name = loss_name

        # handle rescale mode
        if not rescale_mode:
            self.rescale_fn = lambda loss, t: loss
        else:
            rescale_fn_name = f'{rescale_mode}_rescale'
            assert hasattr(self, rescale_fn_name)
            if rescale_mode == 'dynamic_weight':
                if sampler is not None and hasattr(sampler, 'weight'):
                    weight = sampler.weight
                else:
                    assert weight is not None and isinstance(
                        weight, torch.Tensor), (
                            '\'weight\' or a \'sampler\' contains weight '
                            'attribute is must be \'torch.Tensor\' for '
                            '\'dynamic_weight\' rescale_mode.')

                mmcv.print_log(
                    'Apply \'dynamic_weight\' rescale_mode for '
                    f'{self._loss_name}. Please make sure the passed weight '
                    'can be updated by external functions.', 'mmgen')

                rescale_cfg = dict(weight=weight)
            self.rescale_fn = partial(
                getattr(self, rescale_fn_name), **rescale_cfg)

    @staticmethod
    def constant_rescale(loss, timesteps, scale):
        return loss * scale

    @staticmethod
    def dynamic_weight_rescale(loss, timesteps, weight, scale=1):
        return loss * weight[timesteps] * scale

    @torch.no_grad()
    def collect_log(self, loss, timesteps):
        if not self.log_fn_list:
            return

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder_l = [torch.zeros_like(loss) for _ in range(ws)]
            placeholder_t = [torch.zeros_like(timesteps) for _ in range(ws)]
            dist.all_gather(placeholder_l, loss)
            dist.all_gather(placeholder_t, timesteps)
            loss = torch.cat(placeholder_l, dim=0)
            timesteps = torch.cat(placeholder_t, dim=0)
        log_vars = dict()

        if (dist.is_initialized()
                and dist.get_rank() == 0) or not dist.is_initialized():
            for log_fn in self.log_fn_list:
                log_vars.update(log_fn(loss, timesteps))
        self.log_vars = log_vars

    @torch.no_grad()
    def quartile_log_collect(self,
                             loss,
                             timesteps,
                             total_timesteps,
                             prefix_name,
                             reduction='mean'):
        quartile = (timesteps / total_timesteps * 4).type(torch.LongTensor)
        log_vars = dict()

        for idx in range(4):
            if not (quartile == idx).any():
                loss_quartile = torch.zeros((1, ))
            else:
                loss_quartile = reduce_loss(loss[quartile == idx], reduction)
            log_vars[f'{prefix_name}_quartile_{idx}'] = loss_quartile.item()

        return log_vars

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for TimeStepPixelWiseLoss, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        # update log_vars of this class
        self.collect_log(loss, timesteps=timesteps)

        loss_rescaled = self.rescale_fn(loss, timesteps)
        return reduce_loss(loss_rescaled, self.reduction)

    def _forward_loss(self, output_dict):
        raise NotImplementedError

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


@MODULES.register_module()
class DDPMVLBLoss(DDPMLoss):
    """Variational lower-bound loss for DDPM-based models.
    In this loss, we calculate VLB of different timesteps with different
    method. Detailly, ``DiscretizedGaussianLogLikelihoodLoss`` is used when
    timesteps = 0 and ``GaussianKLDLoss`` at other timesteps.
    To control the data flow for loss calculation, users should define
    ``data_info`` and ``data_info_t_0`` for ``GaussianKLDLoss`` and
    ``DiscretizedGaussianLogLikelihoodLoss`` respectively. If not passed
    ``_default_data_info`` and ``_default_data_info_t_0`` would be used.
    To be noted that, we only penalize variance in this loss term, and tensor
    in output dict cooresponding to mean would be detached.

    We additional support another log collection function called
    ``name_log_collection``. In this collection method, we would directly
    collect loss terms calculated by different methods.
    To use this collection methods, users may passed ``log_cfgs`` as the
    following example:

    .. code-block:: python
        :linenos:

        log_cfgs = [
            dict(type='name', reduction=REUCTION, prefix_name=PREFIX),
            ...
        ]

    Args:
        rescale_mode (str, optional): Mode of the loss rescale method.
            Defaults to ''.
        rescale_cfg (dict, optional): Config of the loss rescale method.
        sampler (object): Weight sampler. Defaults to None.
        weight (torch.Tensor, optional): Weight used for rescale losses.
            Defaults to None.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary for ``timesteps != 0``.
            Defaults to None.
        data_info_t_0 (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary for ``timesteps == 0``.
            Defaults to None.
        log_cfgs (list[dict] | dict | optional): Configs to collect logs.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        loss_name (str, optional): Name of the loss item. Defaults to
            'loss_ddpm_vlb'.
    """
    _default_data_info = dict(
        mean_pred='mean_pred',
        mean_target='mean_target',
        logvar_pred='logvar_pred',
        logvar_target='logvar_target')
    _default_data_info_t_0 = dict(
        x='real_imgs', mean='mean_pred', logvar='logvar_pred')

    def __init__(self,
                 rescale_mode='',
                 rescale_cfg=None,
                 sampler=None,
                 weight=None,
                 data_info=None,
                 data_info_t_0=None,
                 log_cfgs=None,
                 reduction='mean',
                 loss_name='loss_ddpm_vlb'):
        super().__init__(log_cfgs, rescale_mode, rescale_cfg, sampler, weight,
                         reduction, loss_name)

        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.reduction = reduction
        self._loss_name = loss_name

        self.data_info = self._default_data_info \
            if data_info is None else data_info
        self.data_info_t_0 = self._default_data_info_t_0 \
            if data_info_t_0 is None else data_info

        self.loss_list = [
            DiscretizedGaussianLogLikelihoodLoss(
                reduction='flatmean',
                data_info=self.data_info_t_0,
                base='2',
                loss_weight=-1,
                only_update_var=True),
            GaussianKLDLoss(
                reduction='flatmean',
                data_info=self.data_info,
                base='2',
                only_update_var=True)
        ]
        self.loss_select_fn_list = [lambda t: t == 0, lambda t: t != 0]

    @torch.no_grad()
    def name_log_collect(self, loss, timesteps, prefix_name, reduction='mean'):
        """Collect loss logs by name (GaissianKLD and
        DiscGaussianLogLikelihood)."""
        log_vars = dict()
        for select_fn, loss_fn in zip(self.loss_select_fn_list,
                                      self.loss_list):
            mask = select_fn(timesteps)
            if not mask.any():
                loss_reduced = torch.zeros((1, ))
            else:
                loss_reduced = reduce_loss(loss[mask], reduction)
            # remove original prefix in loss names
            loss_term_name = loss_fn.loss_name().replace('loss_', '')
            log_vars[f'{prefix_name}_{loss_term_name}'] = loss_reduced.item()

        return log_vars

    def _forward_loss(self, outputs_dict):
        # use `zeros` instead of `zeros_like` to avoid get int tensor
        timesteps = outputs_dict['timesteps']
        loss = torch.zeros(*timesteps.shape).to(timesteps.device)
        for select_fn, loss_fn in zip(self.loss_select_fn_list,
                                      self.loss_list):
            mask = select_fn(timesteps)
            outputs_dict_ = {}
            for k, v in outputs_dict.items():
                if v is None or not isinstance(v, (torch.Tensor, list)):
                    outputs_dict_[k] = v
                else:
                    outputs_dict_[k] = v[mask]
            loss[mask] = loss_fn(outputs_dict_)
        return loss


@MODULES.register_module()
class DDPMMSELoss(DDPMLoss):
    """Mean square loss for DDPM-based models.

    Args:
        rescale_mode (str, optional): Mode of the loss rescale method.
            Defaults to ''.
        rescale_cfg (dict, optional): Config of the loss rescale method.
        sampler (object): Weight sampler. Defaults to None.
        weight (torch.Tensor, optional): Weight used for rescale losses.
            Defaults to None.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary for ``timesteps != 0``.
            Defaults to None.
        log_cfgs (list[dict] | dict | optional): Configs to collect logs.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        loss_name (str, optional): Name of the loss item. Defaults to
            'loss_ddpm_vlb'.
    """
    _default_data_info = dict(pred='eps_t_pred', target='noise')

    def __init__(self,
                 rescale_mode='',
                 rescale_cfg=None,
                 sampler=None,
                 weight=None,
                 log_cfgs=None,
                 reduction='mean',
                 data_info=None,
                 loss_name='loss_ddpm_mse'):
        super().__init__(log_cfgs, rescale_mode, rescale_cfg, sampler, weight,
                         reduction, loss_name)

        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.reduction = reduction
        self._loss_name = loss_name

        self.data_info = self._default_data_info \
            if data_info is None else data_info

        self.loss_fn = partial(mse_loss, reduction='flatmean')

    def _forward_loss(self, outputs_dict):
        loss_input_dict = {
            k: outputs_dict[v]
            for k, v in self.data_info.items()
        }
        loss = self.loss_fn(**loss_input_dict)
        return loss
