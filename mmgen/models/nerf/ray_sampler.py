import numpy as np
import torch
import torch.nn.functional as F

from mmgen.models.builder import MODULES
from .util import sample_in_range


@MODULES.register_module('FullRaySampler')
@MODULES.register_module()
class RaySampler(object):
    """Sample points from on a plane, and the plane is in image coordinates,
    and will be trans to camera coordinates in ``Camera`` class."""

    def __init__(self, H_range, W_range, n_points=None, is_homo=True):
        self.H_range, self.W_range = H_range, W_range
        self.n_points = n_points
        self.training = True
        self.is_homo = is_homo

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def sample_plane_full(self, batch_size=1):
        """Get the index of the full image plane.

        Args:
            batch_size (int): The batch size of the plane we want to generate.
                Defaults to 1.
        """
        # get a full plane
        H = self.H_range[1] - self.H_range[0]
        W = self.W_range[1] - self.W_range[0]
        x, y = torch.meshgrid(
            torch.linspace(self.W_range[0], self.W_range[1] - 1, W),
            torch.linspace(self.H_range[0], self.H_range[1] - 1, H))

        x = x.T.flatten()
        y = y.T.flatten()
        total_points = x.shape[0]
        plane = torch.ones(total_points, 4) if self.is_homo else \
            torch.ones(total_points, 3)
        plane[:, 0] = x
        plane[:, 1] = y

        # repeat at the first dimension
        return plane.unsqueeze(0).repeat([batch_size, 1, 1])

    def select_index(self, plane):
        """
        Args:
            planne (torch.Tensor): [N, 3] or [N, 4]
            n_poinst (int, optional): If None, sample all points

        Returns:
            np.ndarray: Selected index
        """
        if not self.training or self.n_points is None:
            return self._select_index_eval(plane)
        return self._select_index_train(plane)

    def sample_points(self, plane, *args):
        """Sample points on plane."""
        if not self.training or self.n_points is None:
            return self._sample_points_eval(plane, *args)
        return self._sample_points_train(plane, *args)

    def sample_pixels(self, image, *args):
        """Sample pixels on image."""
        # pixels = image.view(3, -1).t()
        # [bz, 3, H, W] to [bz, H*W, 3]
        batch_size = image.shape[0]
        pixels = image.clone().permute(0, 2, 3, 1).view(batch_size, -1, 3)
        if not self.training or self.n_points is None:
            return self._sample_pixels_eval(pixels, *args)
        return self._sample_pixels_train(pixels, *args)

    def _select_index_train(self, plane):
        """
        Args:
            plane (torch.Tensor)

        Returns:
            np.ndarray: Selected index for training.
        """

        batch_size, total_points = plane.shape[:2]
        selected_idx = sample_in_range(total_points, self.n_points, batch_size)
        return selected_idx

    def _select_index_eval(self, plane):
        batch_size, total_points = plane.shape[:2]
        selected_idx = np.repeat(
            np.arange(total_points)[None, ...], batch_size, axis=0)
        return selected_idx

    def _sample_points_train(self, plane, index):
        """Sample plane in when self.training is True."""
        # same usage as `torch.gather`
        return np.take_along_axis(plane, index[..., None], axis=1)

    def _sample_points_eval(self, plane, *args):
        """Sample plane when self.training is False."""
        return plane

    def _sample_pixels_train(self, image, index):
        """
        Args:
            image (torch.Tensor)
            index (np.ndarray)
        Returns:
            torch.Tensor
        """
        # same usage as `torch.gather`
        return np.take_along_axis(image, index[..., None], axis=1)

    def _sample_pixels_eval(self, image, *args):
        return image

    def vis_pixels_on_image(self, image, points, alpha=None):
        """This function use to vis selected points on image. Generate a alpha
        mask.
        Args:
            image (torch.Tensor)
            points (torch.Tensor): The coordinates on the image plane, range in
                [0, H] and [0, W]. We use this to generate mask.
            alpha (bool): Generate a alpha mask. This is useful when selected
                pixels are grouped together (e.g. points generated by
                ``FlexGridSampler``). Defaults to False.

        Returns:
            dict: A dict contains alpha and mask.

        """
        mask = torch.zeros_like(image)
        x_points = points[:, 0, ...].flatten().T.type(torch.LongTensor)
        y_points = points[:, 1, ...].flatten().T.type(torch.LongTensor)
        mask[:, x_points, y_points] = 1
        pixels = image * mask

        vis_dict = dict(vis_mask=mask, vis_pixels=pixels)
        if alpha is not None:
            canvas = image * (1 - alpha) + (1 - mask) * alpha
            vis_dict['vis_alpht'] = canvas

        return vis_dict

    def sample_rays(self, batch_size=1, image=None, *args, **kwargs):
        """
        Args:
            plane: [N, 3] or [N, 4]
            image: [3, H, W]
            batch_size (int, optional): Batch size. If ``image`` is given,
                batch size of ``image`` must consistency with ``batch_size``.
                Defaults to 1.
            we assert H*W == N
        """
        if image is not None:
            assert image.shape[0] == batch_size, (
                'Batch size of the given image must be consistency with '
                f'\'batch_size\', but receive \'{image.shape[0]}\' and '
                f'\'{batch_size}\'')

        # init plane
        plane = self.sample_plane_full(batch_size)

        sample_dict = dict()
        selected_idx = self.select_index(plane)
        points_selected = self.sample_points(plane, selected_idx)

        sample_dict = dict(
            selected_idx=selected_idx, points_selected=points_selected)

        if image is not None:
            # [bz, H*W, 3] --> [bz, n_points, 3]
            image_selected = self.sample_pixels(image, selected_idx)
            sample_dict['real_pixels'] = image_selected

        return sample_dict


@MODULES.register_module()
class PrecropRaySampler(RaySampler):

    def __init__(self, precrop_frac, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precrop_frac = self.curr_precrop_frac = precrop_frac

    def set_precrop_frac(self, precrop_frac):
        """Interface for Hook to set ``curr_precrop_frac``
        Args:
            precrop_frac (float): Fraction of the preprocess crop operation of
                the current iteration.
        """
        self.curr_precrop_frac = precrop_frac

    def _get_index_to_select(self, plane):
        """Get index to select under ``self.curr_percrop_frac``
        Args:
            plane (torch.Tensor): [bz, H*W, 3/4]

        Returns:
            np.ndarray: Index under the current precrop parameters, shape
                like [bz, N'].
        """
        H = self.H_range[1] - self.H_range[0]
        W = self.W_range[1] - self.W_range[0]
        precrop_frac = self.curr_precrop_frac

        dH = int(H // 2 * precrop_frac)
        dW = int(W // 2 * precrop_frac)
        H_cent = (self.H_range[0] + self.H_range[1]) // 2
        W_cent = (self.W_range[0] + self.W_range[1]) // 2

        H_range_percrop = (H_cent - dH, H_cent + dH - 1)
        W_range_percrop = (W_cent - dW, W_cent + dW - 1)

        plane_ = plane[0]  # get a single plane shape like [H*W, 3/4]
        mask_H = torch.logical_and(plane_[..., 0] >= H_range_percrop[0],
                                   plane_[..., 0] <= H_range_percrop[1])
        mask_W = torch.logical_and(plane_[..., 1] >= W_range_percrop[0],
                                   plane_[..., 1] <= W_range_percrop[1])
        mask = torch.logical_and(mask_H, mask_W)  # [N, ]
        index_to_select = torch.nonzero(mask).squeeze()  # [1, N']

        # expand to [bz, N']
        index_to_select = index_to_select.repeat([plane.shape[0], 1])
        return index_to_select.numpy()

    def _select_index_train(self, plane):
        """
        Args:
            plane (np.ndarray): shape like [bz, H*W, 3/4]

        Returns:
            np.ndarray: Selected index in the full image space, shape like
                `[bz, N]`.
        """
        index_to_select = self._get_index_to_select(plane)

        batch_size, total_points = plane.shape[0], index_to_select.shape[1]

        selected_idx = np.take_along_axis(
            index_to_select,
            sample_in_range(total_points, self.n_points, batch_size),
            axis=1)
        return selected_idx

    def _select_index_eval(self, plane):
        """
        Args:
            plane (np.ndarray): shape like [bz, H*W, 3/4]

        Returns:
            np.ndarray: All index under current precrop parameter, shape like
                `[bz, N]`.
        """

        index_to_select = self._get_index_to_select(plane)
        return index_to_select

    def _sample_points_eval(self, plane, index):
        """We only use points under precrop for evaluation.

        Therefore we use the same sample method as
        ``self._sample_points_train``
        """
        return self._sample_points_train(plane, index)

    def _sample_pixels_eval(self, image, index):
        """We only use points under precrop for evaluation.

        Therefore we use the same sample method as
        ``self._sample_pixels_train``
        """
        return self._sample_pixels_train(image, index)


@MODULES.register_module()
class FlexGridRaySampler(RaySampler):

    def __init__(self, min_scale, max_scale, random_shift, random_scale, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.N_samples_sqrt = int(np.sqrt(self.n_points))
        self.random_scale = random_scale
        self.random_shift = random_shift
        self.min_scale = self.curr_min_scale = min_scale
        self.max_scale = max_scale
        self.base_coord = self._get_base_coord()

    def set_min_scale(self, min_scale):
        """Interface for Hook to set ``curr_min_scale``.

        Args:
            curr_min_scale (float): The minize scale for the current iteration.
        """
        self.curr_min_scale = min_scale

    def _get_base_coord(self):
        """Plane based on num points, then apply rescale.

        Returns:
            torch.Tensor: The stack of w and h coordinates, shape as
                `[N_samples_sqrt, N_samples_sqrt, 2]`.
        """
        w, h = torch.meshgrid([
            torch.linspace(-1, 1, self.N_samples_sqrt),
            torch.linspace(-1, 1, self.N_samples_sqrt)
        ])
        coords = torch.cat([w[..., None], h[..., None]], dim=2)
        return coords

    def _prepare_for_grid_sample(self, tensor):
        """Input a flatten tensor shape as [bz, H*W, n], and unflattent it to.

        [bz, n, H, W] for grid sample

        Args:
            tensor (torch.Tensor): Tensor to unflatten.

        Returns:
            torch.Tensor: The unflattened tensor.
        """
        batch_size = tensor.shape[0]
        tensor = tensor.view(batch_size, self.H_range[1] - self.H_range[0],
                             self.W_range[1] - self.W_range[0], -1)
        tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def _prepare_for_camera(self, tensor):
        """Flatten the input tensor from ``[bz, n, H, W]`` to [bz, H*W, n] for
        the camera operation.

        Args:
            tensor (torch.Tensor): Tensor to flatten.

        Returns:
            torch.Tensor: The flattened tensor.
        """
        batch_size, num_chn, H, W = tensor.shape
        tensor = tensor.view(batch_size, num_chn, H * W).permute(0, 2, 1)
        return tensor

    def _select_index_train(self, plane, *args):
        """Return relatively coordinates, range in [-1, 1]
        Args:
            plane (torch.Tensor): shape like [bz, ]

        Returns:
            torch.Tensor: The sampled relative coordinates.
        """
        batch_size = plane.shape[0]

        base_plane = torch.cat(
            [self.base_coord[None, ...]] * batch_size, dim=0)
        # '-1' because we will slice base_plane
        target_shape = [1 for _ in range(base_plane.ndim - 1)]
        target_shape[0] = batch_size

        if self.random_scale:
            scale = np.random.uniform(
                self.curr_min_scale, self.max_scale, size=target_shape)
            base_plane = base_plane
        else:
            scale = 1

        if self.random_shift:
            max_offset = 1 - scale
            h_offset = np.random.uniform(
                0, max_offset, size=target_shape) * (
                    np.random.randint(2, size=target_shape) - 0.5) * 2
            w_offset = np.random.uniform(
                0, max_offset, size=target_shape) * (
                    np.random.randint(2, size=target_shape) - 0.5) * 2
            base_plane[..., 0] += h_offset
            base_plane[..., 1] += w_offset

        return base_plane

    def _sample_points_train(self, plane, coords):
        """Sample points on plane with given coordinates.
        ``torch.nn.functional.grid_sample`` is used because the coordinates are
        not integer.

        Args:
            plane (torch.Tensor): The plane to be sampled. Shape like
                ``[bz, H, W, 3/4]``.
            coords (torch.Tensor): The relative coordinates of points to be
                sampled. Shape like
                ``[bz, N_samples_sqrt, N_samples_sqrt, 2]``.

        Returns:
            torch.Tensor: The coordinates of the sampled points on the image
                plane. Shape like
                ``[bz, N_samples_sqrt*N_samples_sqrt, 3/4]``.
        """
        # plane: [bz, H*W, 3/4] to [bz, 3/4, H, W] for grid sample
        plane_unflatten = self._prepare_for_grid_sample(plane)

        # coords: [bz, H', W', 2]
        points_selected = F.grid_sample(
            plane_unflatten, coords, mode='bilinear', align_corners=True)

        return self._prepare_for_camera(points_selected)

    def _sample_pixels_train(self, image, coords):
        """Sample pixels on image with given coordinates.
        ``torch.nn.functional.grid_sample`` is used because the coordinates are
        not integer.

        Args:
            plane (torch.Tensor): The plane to be sampled. Shape like
                ``[bz, H, W, 3/4]``.
            coords (torch.Tensor): The relative coordinates of points to be
                sampled. Shape like
                ``[bz, N_samples_sqrt, N_samples_sqrt, 2]``.

        Returns:
            torch.Tensor: The pixel values of the sampled points of the given
                image. Shape like ``[bz, N_samples_sqrt*N_samples_sqrt, 3]``.
        """
        # image: [bz, H*W, 3] to [bz, 3, H, W] for grid sample
        image_unflatten = self._prepare_for_grid_sample(image)

        pixels_selected = F.grid_sample(
            image_unflatten.cpu(), coords, mode='bilinear', align_corners=True)
        return self._prepare_for_camera(pixels_selected)
