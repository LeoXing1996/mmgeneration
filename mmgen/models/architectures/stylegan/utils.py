# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmgen.ops.stylegan3.ops import upfirdn2d
from ..common import get_module_device


@torch.no_grad()
def get_mean_latent(generator, num_samples=4096, bs_per_repeat=1024):
    """Get mean latent of W space in Style-based GANs.

    Args:
        generator (nn.Module): Generator of a Style-based GAN.
        num_samples (int, optional): Number of sample times. Defaults to 4096.
        bs_per_repeat (int, optional): Batch size of noises per sample.
            Defaults to 1024.

    Returns:
        Tensor: Mean latent of this generator.
    """
    device = get_module_device(generator)
    mean_style = None
    n_repeat = num_samples // bs_per_repeat
    assert n_repeat * bs_per_repeat == num_samples

    for _ in range(n_repeat):
        style = generator.style_mapping(
            torch.randn(bs_per_repeat,
                        generator.style_channels).to(device)).mean(
                            0, keepdim=True)
        if mean_style is None:
            mean_style = style
        else:
            mean_style += style
    mean_style /= float(n_repeat)

    return mean_style


@torch.no_grad()
def style_mixing(generator,
                 n_source,
                 n_target,
                 inject_index=1,
                 truncation_latent=None,
                 truncation=0.7,
                 style_channels=512,
                 **kwargs):
    """Generating style mixing images.

    Args:
        generator (nn.Module): Generator of a Style-Based GAN.
        n_source (int): Number of source images.
        n_target (int): Number of target images.
        inject_index (int, optional): Index from which replace with source
            latent. Defaults to 1.
        truncation_latent (torch.Tensor, optional): Mean truncation latent.
            Defaults to None.
        truncation (float, optional): Truncation factor. Give value less
            than 1., the truncation trick will be adopted. Defaults to 1.
        style_channels (int): The number of channels for style code.

    Returns:
        torch.Tensor: Table of style-mixing images.
    """
    device = get_module_device(generator)
    source_code = torch.randn(n_source, style_channels).to(device)
    target_code = torch.randn(n_target, style_channels).to(device)

    source_image = generator(
        source_code,
        truncation_latent=truncation_latent,
        truncation=truncation,
        **kwargs)

    h, w = source_image.shape[-2:]
    images = [torch.ones(1, 3, h, w).to(device) * -1]

    target_image = generator(
        target_code,
        truncation_latent=truncation_latent,
        truncation=truncation,
        **kwargs)

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            truncation_latent=truncation_latent,
            truncation=truncation,
            inject_index=inject_index,
            **kwargs)
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)

    return images


def apply_integer_translation(x, tx, ty):
    """Apply integer offset translation to feature map.

    Args:
        x (torch.Tensor): Intermediate feature map.
        tx (float): X-axis translation to image width ratio.
        ty (float): Y-axis translation to image height ratio.

    Returns:
        torch.Tensor: Feature map after translation.
    """
    _N, _C, H, W = x.shape
    tx = torch.as_tensor(tx * W).to(dtype=torch.float32, device=x.device)
    ty = torch.as_tensor(ty * H).to(dtype=torch.float32, device=x.device)
    ix = tx.round().to(torch.int64)
    iy = ty.round().to(torch.int64)

    z = torch.zeros_like(x)
    m = torch.zeros_like(x)
    if abs(ix) < W and abs(iy) < H:
        y = x[:, :, max(-iy, 0):H + min(-iy, 0), max(-ix, 0):W + min(-ix, 0)]
        z[:, :, max(iy, 0):H + min(iy, 0), max(ix, 0):W + min(ix, 0)] = y
        m[:, :, max(iy, 0):H + min(iy, 0), max(ix, 0):W + min(ix, 0)] = 1
    return z, m


def sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)


def apply_fractional_translation(x, tx, ty, a=3):
    """Apply fractional offset translation to feature map.

    Args:
        x (torch.Tensor): Intermediate feature map.
        tx (float): X-axis translation to image width ratio.
        ty (float): Y-axis translation to image height ratio.
        a (int): Spatial extent of the Lanczos kernel. Defaults to 3.

    Returns:
        torch.Tensor: Feature map after translation.
    """
    _N, _C, H, W = x.shape
    tx = torch.as_tensor(tx * W).to(dtype=torch.float32, device=x.device)
    ty = torch.as_tensor(ty * H).to(dtype=torch.float32, device=x.device)
    ix = tx.floor().to(torch.int64)
    iy = ty.floor().to(torch.int64)
    fx = tx - ix
    fy = ty - iy
    b = a - 1

    z = torch.zeros_like(x)
    zx0 = max(ix - b, 0)
    zy0 = max(iy - b, 0)
    zx1 = min(ix + a, 0) + W
    zy1 = min(iy + a, 0) + H
    if zx0 < zx1 and zy0 < zy1:
        taps = torch.arange(a * 2, device=x.device) - b
        filter_x = (sinc(taps - fx) * sinc((taps - fx) / a)).unsqueeze(0)
        filter_y = (sinc(taps - fy) * sinc((taps - fy) / a)).unsqueeze(1)
        y = x
        y = upfirdn2d.filter2d(
            y, filter_x / filter_x.sum(), padding=[b, a, 0, 0])
        y = upfirdn2d.filter2d(
            y, filter_y / filter_y.sum(), padding=[0, 0, b, a])
        y = y[:, :,
              max(b - iy, 0):H + b + a + min(-iy - a, 0),
              max(b - ix, 0):W + b + a + min(-ix - a, 0)]
        z[:, :, zy0:zy1, zx0:zx1] = y

    m = torch.zeros_like(x)
    mx0 = max(ix + a, 0)
    my0 = max(iy + a, 0)
    mx1 = min(ix - b, 0) + W
    my1 = min(iy - b, 0) + H
    if mx0 < mx1 and my0 < my1:
        m[:, :, my0:my1, mx0:mx1] = 1
    return z, m


def rotation_matrix(angle):
    """Get rotation matrix.

    Args:
        angle (float): Rotation angle.

    Returns:
        torch.Tensor: Rotation matrix.
    """
    angle = torch.as_tensor(angle).to(torch.float32)
    mat = torch.eye(3, device=angle.device)
    mat[0, 0] = angle.cos()
    mat[0, 1] = angle.sin()
    mat[1, 0] = -angle.sin()
    mat[1, 1] = angle.cos()
    return mat


def lanczos_window(x, a):
    """Get 2D lanczos kernel.

    Args:
        x (torch.Tensor):
        a (int): Spatial extent of the Lanczos kernel. Defaults to 3.

    Returns:
        torch.Tensor: Lanczos window.
    """
    x = x.abs() / a
    return torch.where(x < 1, sinc(x), torch.zeros_like(x))


def construct_affine_bandlimit_filter(mat,
                                      a=3,
                                      amax=16,
                                      aflt=64,
                                      up=4,
                                      cutoff_in=1,
                                      cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up -
            aflt).roll(1 - aflt * up)
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fin = sinc(xi * cutoff_in) * sinc(yi * cutoff_in)
    fout = sinc(xo * cutoff_out) * sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fin) * torch.fft.fftn(fout)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = lanczos_window(xi, a) * lanczos_window(yi, a)
    wo = lanczos_window(xo, a) * lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0, 1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f,
                                [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2,
                                                      up)
    f = f / f.sum([0, 2], keepdim=True) / (up**2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f


def apply_affine_transformation(x, mat, up=4, **filter_kwargs):
    """Apply affine transformation to feature map.

    Args:
        x (torch.Tensor): Intermediate feature map.
        mat (float): Rotation matrix.
        up (int): Upsampling factor.

    Returns:
        torch.Tensor: Feature map after translation.
    """
    _N, _C, H, W = x.shape
    mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

    # Construct filter.
    f = construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
    assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
    p = f.shape[0] // 2

    # Construct sampling grid.
    theta = mat.inverse()
    theta[:2, 2] *= 2
    theta[0, 2] += 1 / up / W
    theta[1, 2] += 1 / up / H
    theta[0, :] *= W / (W + p / up * 2)
    theta[1, :] *= H / (H + p / up * 2)
    theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
    g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

    # Resample image.
    y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
    z = torch.nn.functional.grid_sample(
        y, g, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Form mask.
    m = torch.zeros_like(y)
    c = p * 2 + 1
    m[:, :, c:-c, c:-c] = 1
    m = torch.nn.functional.grid_sample(
        m, g, mode='nearest', padding_mode='zeros', align_corners=False)
    return z, m


def apply_fractional_rotation(x, angle, a=3, **filter_kwargs):
    """Apply fractional rotation to feature map.

    Args:
        x (torch.Tensor): Intermediate feature map.
        angle (float): Rotate angle.
        a (int): Spatial extent of the Lanczos kernel. Defaults to 3.

    Returns:
        torch.Tensor: Feature map after rotation.
    """
    angle = torch.as_tensor(angle).to(dtype=torch.float32, device=x.device)
    mat = rotation_matrix(angle)
    return apply_affine_transformation(
        x, mat, a=a, amax=a * 2, **filter_kwargs)


def apply_fractional_pseudo_rotation(x, angle, a=3, **filter_kwargs):
    angle = torch.as_tensor(angle).to(dtype=torch.float32, device=x.device)
    mat = rotation_matrix(-angle)
    f = construct_affine_bandlimit_filter(
        mat, a=a, amax=a * 2, up=1, **filter_kwargs)
    y = upfirdn2d.filter2d(x=x, f=f)
    m = torch.zeros_like(y)
    c = f.shape[0] // 2
    m[:, :, c:-c, c:-c] = 1
    return y, m
