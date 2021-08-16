import warnings
from copy import deepcopy
from functools import partial

import mmcv
import torch
import torch.nn.functional as F
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ExponentialMovingAverageHook(Hook):
    """Exponential Moving Average Hook.

    Exponential moving average is a trick that widely used in current GAN
    literature, e.g., PGGAN, StyleGAN, and BigGAN. This general idea of it is
    maintaining a model with the same architecture, but its parameters are
    updated as a moving average of the trained weights in the original model.
    In general, the model with moving averaged weights achieves better
    performance.

    Args:
        module_keys (str | tuple[str]): The name of the ema model. Note that we
            require these keys are followed by '_ema' so that we can easily
            find the original model by discarding the last four characters.
        interp_mode (str, optional): Mode of the interpolation method.
            Defaults to 'lerp'.
        interp_cfg (dict | None, optional): Set arguments of the interpolation
            function. Defaults to None.
        interval (int, optional): Evaluation interval (by iterations).
            Default: -1.
        start_iter (int, optional): Start iteration for ema. If the start
            iteration is not reached, the weights of ema model will maintain
            the same as the original one. Otherwise, its parameters are updated
            as a moving average of the trained weights in the original model.
            Default: 0.
    """

    def __init__(self,
                 module_keys,
                 interp_mode='lerp',
                 interp_cfg=None,
                 interval=-1,
                 update_sn=False,
                 start_iter=0):
        super().__init__()
        assert isinstance(module_keys, str) or mmcv.is_tuple_of(
            module_keys, str)
        self.module_keys = (module_keys, ) if isinstance(module_keys,
                                                         str) else module_keys
        # sanity check for the format of module keys
        for k in self.module_keys:
            assert k.endswith(
                '_ema'), 'You should give keys that end with "_ema".'
        self.interp_mode = interp_mode
        self.interp_cfg = dict() if interp_cfg is None else deepcopy(
            interp_cfg)
        self.interval = interval
        self.start_iter = start_iter
        self.update_sn = update_sn

        assert hasattr(
            self, interp_mode
        ), f'Currently, we do not support {self.interp_mode} for EMA.'
        self.interp_func = partial(
            getattr(self, interp_mode), **self.interp_cfg)

    @staticmethod
    def lerp(a,
             b,
             k,
             momentum=0.999,
             momentum_nontrainable=0.,
             trainable=True):
        m = momentum if trainable else momentum_nontrainable
        return a + (b - a) * m

    @staticmethod
    def prefix_lerp(a,
                    b,
                    k,
                    momentum=0.999,
                    momentum_nontrainable=0.,
                    trainable=True,
                    prefix_momentum_dict=dict()):
        m = None
        for prefix, m_ in prefix_momentum_dict.items():
            if prefix in k:
                m = m_
                break
        if m is None:
            m = momentum if trainable else momentum_nontrainable
        return a + (b - a) * m

    def every_n_iters(self, runner, n):
        if runner.iter < self.start_iter:
            return True
        return (runner.iter + 1 - self.start_iter) % n == 0 if n > 0 else False

    @torch.no_grad()
    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model

        for key in self.module_keys:
            # get current ema states
            ema_net = getattr(model, key)
            states_ema = ema_net.state_dict(keep_vars=False)
            # get currently original states
            net = getattr(model, key[:-4])
            states_orig = net.state_dict(keep_vars=True)

            for k, v in states_orig.items():
                if runner.iter < self.start_iter:
                    states_ema[k].data.copy_(v.data)
                else:
                    states_ema[k] = self.interp_func(
                        v, states_ema[k], k,
                        trainable=v.requires_grad).detach()
            if runner.iter >= self.start_iter and self.update_sn:
                # import time
                # s_time = time.time()
                # update_v(states_ema)
                # e_time = time.time()
                # print(f'update v cost: {e_time - s_time}')
                update_v(states_ema, fast=True)
            ema_net.load_state_dict(states_ema, strict=True)

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model
        # sanity check for ema model
        for k in self.module_keys:
            if not hasattr(model, k) and not hasattr(model, k[:-4]):
                raise RuntimeError(
                    f'Cannot find both {k[:-4]} and {k} network for EMA hook.')
            if not hasattr(model, k) and hasattr(model, k[:-4]):
                setattr(model, k, deepcopy(getattr(model, k[:-4])))
                warnings.warn(
                    f'We do not suggest construct and initialize EMA model {k}'
                    ' in hook. You may explicitly define it by yourself.')


# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
    """Orthogonalize x wrt list of vectors ys."""
    for y in ys:
        x = x - proj(x, y)
    return x


def power_iteration(W, u_, update=True, eps=1e-12):
    """Apply num_itrs steps of the power method to estimate top N singular
    values."""
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other
            # singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other
            # singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us, vs


def solve_v_and_rescale(weight_mat, u, target_sigma):
    # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
    # (the invariant at top of this class) and `u @ W @ v = sigma`.
    # This uses pinverse in case W^T W is not invertible.
    # v = torch.linalg.multi_dot([
    #     weight_mat.t().mm(weight_mat).pinverse(),
    #     weight_mat.t(),
    #     u.unsqueeze(1)
    # ]).squeeze(1)
    # v = ((weight_mat.t() @ weight_mat).pinverse() @ weight_mat.t()
    #      @ u.unsqueeze(1)).squeeze()

    v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(),
                           weight_mat.t(), u.unsqueeze(1)).squeeze(1)
    return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))


def update_v(state_dict, fast=False):
    for m in state_dict.keys():
        # TODO: may be a more elegant way
        m_list = m.split('.')
        if 'weight_v' in m_list:
            weight = state_dict[m.replace('_v', '_orig')]
            u = state_dict[m.replace('_v', '_u')]
            weight = weight.reshape(weight.size(0), -1)
            if not fast:
                svs, us, vs = power_iteration(
                    weight.clone(), deepcopy([u.unsqueeze(0)]), update=False)
                v = solve_v_and_rescale(weight, u, svs[0])
                # print(f'Diff [v - vs] {torch.abs(v - vs[0]).max()}')
                # print(f'Diff [v - v old] {torch.abs(v - v_old).max()}')
            else:
                from torch.nn.functional import normalize
                v = state_dict[m]
                v = normalize(torch.mv(weight.t(), u), dim=0, eps=1e-8, out=v)
                u = normalize(torch.mv(weight, v), dim=0, eps=1e-8, out=u)
                state_dict[m.replace('_v', '_u')].data.copy_(u)
            state_dict[m].data.copy_(v)
