# Copyright (c) OpenMMLab. All rights reserved.
import functools

import numpy as np
import torch
from mmcv.utils import is_list_of


def inverse_transform_sampling(bins,
                               weights,
                               N_samples,
                               det=False,
                               device='cpu'):
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)

    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # np.random.seed(0)
        # new_shape = list(cdf.shape[:-1]) + [N_samples]
        # u = torch.from_numpy(np.random.rand(*new_shape).astype(np.float32))
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous().to(device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def rescaled_samples(sampling_fn):

    @functools.wraps(sampling_fn)
    def wrapper(num_batches,
                val=None,
                lower_bound=None,
                upper_bound=None,
                offset=None,
                scale=None):
        if val is not None:
            return torch.FloatTensor([val]).repeat(num_batches).unsqueeze(1)

        x = sampling_fn(num_batches)
        if lower_bound is not None and upper_bound is not None:
            assert (lower_bound <= upper_bound).all(), (
                '\'lower_bound\' must less than or equal to \'upper_bound\', '
                f'but receive {lower_bound} and {upper_bound} respectively.')
            x = x * (upper_bound - lower_bound) + lower_bound
        elif offset is not None and scale is not None:
            assert (scale > 0).all(), ('\'scale\' must larger than 0, but '
                                       f'receive {scale}.')
            if 'uniform' in sampling_fn.__name__:
                # rescale [0, 1] to [-1, 1]
                x = (x - 0.5) * 2
            x = x * scale + offset
        else:
            raise ValueError(
                'One and only one of the two sets of parameters, ('
                'lower_bound, upper_bound) and (offset, scale), should not be '
                'None.')
        return x

    return wrapper


@rescaled_samples
def uniform_sampling(num_batches, same_cross_batch=False):
    """"""
    if same_cross_batch:
        return torch.rand((1, )).repeat(num_batches)
    return torch.rand((num_batches, ))


@rescaled_samples
def gaussian_sampling(num_batches, same_cross_batch=False):
    if same_cross_batch:
        return torch.randn((1, )).repeat(num_batches)
    return torch.randn((num_batches, ))


def normalize_vector(vector):
    """Normalize the input vector to a unit vector.
    Args:
        vector (torch.Tensor): The input vector.

    Returns
        torch.Tensor: The unit vector after normalization.
    """
    return vector / torch.norm(vector, dim=-1, keepdim=True)


# TODO: change name
def pose_to_tensor(poses):
    """Check whether the given poses is legal.

    In `Camera`, we support
    poses input as follow :
    1. list / np.ndarray / torch.Tensor shape as [3] or [4]
    2. list / np.ndarray / torch.Tensor shape as [N, 3] or [N, 4]
    """
    if isinstance(poses, list):
        if is_list_of(poses, int):
            # single pose list
            assert len(poses) in [3, 4]
        elif is_list_of(poses, list):
            assert all([len(p) in [3, 4] for p in poses])
        else:
            raise ValueError('Only support list consist with int and shape as '
                             '[3], [4], [N, 3] or [N, 4]. But receive '
                             f'{poses}.')
        return torch.from_numpy(np.array(poses).astype(np.float32))
    elif isinstance(poses, np.ndarray):
        if poses.ndim == 1:
            assert poses.shape[0] in [3, 4]
        elif poses.ndim == 2:
            assert all([p.shape[1] in [3, 4] for p in poses])
        else:
            raise ValueError('Only support np.ndarray shape as [3], [4], '
                             '[N, 3] or [N, 4]. But receive poses shape as '
                             f'{poses.shape}.')
        return torch.from_numpy(poses.astype(np.float32))
    elif isinstance(poses, torch.Tensor):
        if poses.ndim == 1:
            assert poses.shape[0] in [3, 4]
        elif poses.ndim == 2:
            assert all([p.shape[1] in [3, 4] for p in poses])
        else:
            raise ValueError('Only support torch.Tensor shape as [3], [4], '
                             '[N, 3] or [N, 4]. But receive poses shape as '
                             f'{poses.shape}.')
        return poses.float()


def degree2radian(degree):
    """Convert degree to radian.
    Args:
        degree (float, np.array, torch.Float): Angle with degree measurement.

    Returns:
        float, np.array, torch.Float: Converted angle with radian measurement.
    """
    return degree / 180 * np.pi


def points_to_homo(points):
    """Convert points with (x, y, z) or (x, y, z, w)
    Args:
        points ()

    Returns:
        torch.Tensor
    """
    if points.ndim == 1:
        points = points.unsqueeze(0)

    assert points.ndim == 2, ''
    assert points.size(1) in [3, 4], 'TODO:'

    n_points = points.size(0)
    homo = torch.eye(4).repeat(n_points, 1, 1)
    homo[..., 0, -1] = points[:, 0]
    homo[..., 1, -1] = points[:, 1]
    homo[..., 2, -1] = points[:, 2]

    if points.size(1) == 4:
        homo[..., 3, -1] = points[:, 3]

    return homo


def prepare_matrix(matrix, is_batch=True, is_homo=True, allow_clip=True):
    """This function make matrix shape as [N, 3, 3] or [N, 4, 4]

    Args:
        allow_clip: Only work when is_homo = True but the matrix is size as
            [..., 4, 4]. If set as True, we will direct remain the first 3x3
            elements of the matrix (also known as the rotation) and the other
            part (translation transformation) will be directly dropped.
            Otherwise, an error will be raised if the first three elements of
            the last column of the matrix are not all zero.

    Returns:
        torch.Tensor: Converted matrix
    """
    matrix = matrix.clone()
    assert matrix.ndim in [2, 3]
    assert matrix.shape[-2] == matrix.shape[-1]
    assert matrix.shape[-1] in [3, 4]

    # convert [x, x] to [N, x, x]
    if is_batch:
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
    else:
        # squeeze matrix and remove 'batch_size' dimension
        assert matrix.ndim == 2 or matrix.shape[0] == 1, (
            'prepare_matrix with \'is_batch=True\' only support input shape '
            'as [1, n, n] or [n, n], but receive matrix shape as '
            f'{matrix.shape}')
        matrix = matrix.squeeze()

    if is_homo:
        if matrix.shape[-1] == 3:
            tmp = torch.eye(4)
            if is_batch:
                tmp = tmp.unsqueeze(0).repeat(matrix.shape[0], 1, 1)
            tmp[..., :3, :3] = matrix
            matrix = tmp
    else:
        if matrix.shape[-1] == 4:
            have_trans = (matrix[..., :3, 3] == 0).all()
            if have_trans:
                assert allow_clip, (
                    'The input matrix is homogeneous, but the first three '
                    'elements of the last column are not all zero. This means '
                    'the input matrix contains translation transform, and '
                    'directly converting the input to a non-homogeneous '
                    'matrix may cause translation transformation missing. '
                    'Please set \'allow_clip\' as True to force conversion.')
            matrix = matrix[..., :3, :3]

    return matrix


def prepare_vector(vector, is_batch=True, is_homo=True, to_matrix=True):
    """Make vector to target shape.

    Args:
        to_matrix (bool): This parameter only work when is_homo = True.
            Convert the vector size as [4, ] or [N, 4] to [4, 1] and [N, 4, 1]
            in order to support matrix multiply with a homogeneous
            transformation matrix.
    """
    vector = vector.clone()
    assert vector.ndim in [1, 2]
    assert vector.shape[-1] in [3, 4]

    if is_batch:
        if vector.ndim == 1:
            vector = vector.unsqueeze(0)
    else:
        # squeeze vector and remove 'batch_size' dimension
        assert vector.ndim == 1 or vector.shape[0] == 1, (
            'prepare_vector with \'is_batch=True\' only support input shape '
            f'as [1, n] or [n, ], but receive matrix shape as {vector.shape}')
        vector = vector.squeeze()

    if is_homo:
        if vector.shape[-1] == 3:
            ones_shape = [vector.shape[0], 1] if is_batch else [
                1,
            ]
            vector = torch.cat([vector, torch.ones(ones_shape)], dim=-1)
        if to_matrix:
            vector = vector.unsqueeze(-1)
    else:
        if vector.shape[-1] == 4:
            vector = vector[..., :3]

    return vector
