import numpy as np
import torch
import torch.nn.functional as F

from mmgen.models.builder import MODULES


@MODULES.register_module('FullRaySampler')
@MODULES.register_module()
class RaySampler(object):
    """Sample points from on a plane, and the plane is in image coordinates,
    and will be trans to camera coordinates in ``Camera`` class."""

    def __init__(self, H_range, W_range, n_points=None):
        self.H_range, self.W_range = H_range, W_range
        self.n_points = n_points
        self.training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def sample_plane_full(self, is_homo=True):
        """Get the index of the full image plane."""
        # get a full plane
        H = self.H_range[1] - self.H_range[0]
        W = self.W_range[1] - self.W_range[0]
        x, y = torch.meshgrid(
            torch.linspace(self.W_range[0], self.W_range[1] - 1, W),
            torch.linspace(self.H_range[0], self.H_range[1] - 1, H))

        x = x.T.flatten()
        y = y.T.flatten()
        total_points = x.shape[0]
        plane = torch.ones(total_points, 4) if is_homo else \
            torch.ones(total_points, 3)
        plane[:, 0] = x
        plane[:, 1] = y
        return plane

    def select_index(self, plane):
        """
        Args:
            planne (torch.Tensor): [N, 3] or [N, 4]
            n_poinst (int, optional): If None, sample all points
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
        # [3, H, W] to [H*W, 3]
        pixels = image.clone().permute(1, 2, 0).view(-1, 3)
        if not self.training or self.n_points is None:
            return self._sample_pixels_eval(pixels, *args)
        return self._sample_pixels_train(pixels, *args)

    def _select_index_train(self, plane):
        np.random.seed(0)
        total_points = plane.shape[0]
        selected_idx = np.random.choice(
            total_points, self.n_points, replace=False)
        return selected_idx

    def _select_index_eval(self, plane):
        total_points = plane.shape[0]
        selected_idx = np.arange(total_points)
        return selected_idx

    def _sample_points_train(self, plane, index):
        """Sample plane in when self.training is True."""
        return plane[index]

    def _sample_points_eval(self, plane, *args):
        """Sample plane when self.training is False."""
        return plane

    def _sample_pixels_train(self, image, index):
        return image[index]

    def _sample_pixels_eval(self, image, *args):
        return image

    def sample_rays(self, image=None):
        """
        Args:
            plane: [N, 3] or [N, 4]
            image: [3, H, W]
            we assert H*W == N
        """
        # init plane
        plane = self.sample_plane_full()

        sample_dict = dict()
        selected_idx = self.select_index(plane)
        points_selected = self.sample_points(plane, selected_idx)

        sample_dict = dict(
            selected_idx=selected_idx, points_selected=points_selected)

        if image is not None:
            # [H*W, 3] --> [n_points, 3]
            image_selected = self.sample_pixels(image, selected_idx)
            sample_dict['real_pixels'] = image_selected

        return sample_dict


@MODULES.register_module()
class PrecropSampler(RaySampler):

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
        """Get index to select under ``self.curr_percrop_frac``"""
        H = self.H_range[1] - self.H_range[0]
        W = self.W_range[1] - self.W_range[0]
        precrop_frac = self.curr_precrop_frac

        dH = int(H // 2 * precrop_frac)
        dW = int(W // 2 * precrop_frac)
        H_cent = (self.H_range[0] + self.H_range[1]) // 2
        W_cent = (self.W_range[0] + self.W_range[1]) // 2

        H_range_percrop = (H_cent - dH, H_cent + dH - 1)
        W_range_percrop = (W_cent - dW, W_cent + dW - 1)

        mask_H = torch.logical_and(plane[:, 0] >= H_range_percrop[0],
                                   plane[:, 0] <= H_range_percrop[1])
        mask_W = torch.logical_and(plane[:, 1] >= W_range_percrop[0],
                                   plane[:, 1] <= W_range_percrop[1])
        index_to_select = torch.nonzero(torch.logical_and(mask_H,
                                                          mask_W)).squeeze()
        return index_to_select

    def _select_index_train(self, plane):
        index_to_select = self._get_index_to_select(plane)

        # NOTE: just for debug
        # np.random.seed(0)

        total_points = index_to_select.shape[0]
        selected_idx = index_to_select[np.random.choice(
            total_points, self.n_points, replace=False)]
        return selected_idx

    def _select_index_eval(self, plane):
        index_to_select = self._get_index_to_select(plane)
        return index_to_select


@MODULES.register_module()
class FlexGridSampler(RaySampler):

    def __init__(self, min_scale, max_scale, random_shift, random_scale, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.N_samples_sqrt = int(np.sqrt(self.num_points))
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
        """Plane based on num points, then apply rescale."""
        w, h = torch.meshgrid([
            torch.linspace(-1, 1, self.N_samples_sqrt),
            torch.linspace(-1, 1, self.N_samples_sqrt)
        ])
        return w, h

    def _select_index_train(self, *args):
        """Return relatively coordinates, range in [-1, 1]"""

        plane = self.base_coord
        if self.random_scale:
            scale = np.random.uniform(self.curr_min_scale, self.max_scale)
            plane = plane * scale
        else:
            scale = 1

        if self.random_shift:
            max_offset = 1 - scale
            h_offset = np.random.uniform(
                0, max_offset) * (np.random.randint(2, (1, )) - 0.5) * 2
            w_offset = np.random.uniform(
                0, max_offset) * (np.random.randint(2, (1, )) - 0.5) * 2
            plane[0] += h_offset
            plane[1] += w_offset

        return plane

    def _sample_points_train(self, plane, coords):
        plane = F.grid_sample(
            plane, coords, mode='bilinear', align_corners=True)
        return plane

    def _sample_pixels_train(self, image, coords):
        pixels_selected = F.grid_sample(
            image, coords, mode='bilinear', align_corners=True)
        return pixels_selected
