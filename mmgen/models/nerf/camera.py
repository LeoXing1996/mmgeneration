# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import torch
from mmcv.utils import is_list_of

from mmgen.models.builder import build_module
from .util import (degree2radian, gaussian_sampling, normalize_vector,
                   pose_to_tensor, prepare_matrix, prepare_vector,
                   uniform_sampling)


class Camera(object):
    r"""
    camera_sample_dist is a dict contains the distribution to sample a random
    camera.

    In this class, we all use homogenerous coordinates

    .. code-block:: python
        :linenons:

    And for ``sample_mode = 'spherical'``, the dict should be like:

    .. code-block:: python
        :linenons:

        Args:
            camera_sample_dist (dict):


    camera_angle_x: angle on x-axis.
    """

    _support_sample_dist = [
        'uniform', 'gaussian', 'normal', 'truncated_gaussian', 'spherical'
    ]

    _default_ray_sampler = dict(type='FullRaySampler', n_points=None)

    def __init__(
            self,
            camera_matrix=None,
            fov=None,
            camera_angle_x=None,
            H=None,
            W=None,
            H_range=None,
            W_range=None,
            ray_sampler=None,
            near=None,
            far=None,
            focal=None,
            u_dist=None,
            v_dist=None,
            theta_dist=None,
            phi_dist=None,
            radius_dist=None,
            camera_sample_mode='uniform',
            # is_hemi_sphere=False,  # clip the sample range
            which_hand='right',
            degree2radian=False):

        self.H, self.W, self.H_range, self.W_range = self.get_image_plane(
            H, W, H_range, W_range)

        # build ray sampler
        if ray_sampler is None:
            ray_sampler = self._default_ray_sampler
        ray_sampler_ = deepcopy(ray_sampler)
        ray_sampler_['H_range'] = self.H_range
        ray_sampler_['W_range'] = self.W_range
        self.ray_sampler = build_module(ray_sampler_)

        self.near, self.far = near, far
        self.degree2radian = degree2radian

        # init instrinst matrix / projection matrix / camera matrix
        self.camera_matrix, self.focal = self.get_camera_matrix(
            camera_matrix, focal, fov, camera_angle_x)

        assert camera_sample_mode in self._support_sample_dist
        self.camera_sample_mode = camera_sample_mode

        # set deterministic input to dict(val=VAL)
        # self.camera_dist = dict()
        # dist_list = [u_dist, v_dist, theta_dist, phi_dist, radius_dist]
        # dist_name_list = ['u', 'v', 'theta', 'phi', 'radius']
        # for name, conf in zip(dist_name_list, dist_list):
        #     if isinstance(conf, dict):
        #         setattr(self, f'{name}_dist', conf)
        #     elif isinstance(conf, int):
        #         setattr(self, f'{name}_dist', dict(val=conf))
        #     else:
        #         print(name, conf)
        #         assert conf is None, 'None or dict or val'

        assert which_hand in ['right', 'left']
        self.which_hand = which_hand
        if self.which_hand == 'right':
            self.up_axis = torch.FloatTensor([0, 1, 0]).unsqueeze(0)
        else:
            self.up_axis = torch.FloatTensor([0, 0, 1]).unsqueeze(0)

    def _handle_single_range(self, edge, edge_range, name='edge'):
        """
        Args:
            edge (int)
            edge_range (list[int])
            name (str, optional): The name of the current edge. Used for error
                message output. Defaults to 'edge'.

        Returns:
            tuple: Tuple contains edge and edge_range.
        """

        edge_name = name
        edge_range_name = f'{name}_range'

        assert (edge is not None) or (edge_range is not None), ('')

        if edge_range is not None:
            assert (len(edge_range) == 2) and is_list_of(edge_range, int), (
                f'\'{edge_range_name}\' must be a list of int and contains '
                f'two elements, but received {edge_range}')
            edge_range.sort()  # make edge to be ordered as [min, max]
            edge_length = edge_range[1] - edge_range[0]

        if edge is None:
            if edge_range is None:
                edge = 2
                edge_range = [-1, 1]
            else:
                edge = edge_length
        else:
            assert isinstance(edge, int)
            if edge_range is None:
                edge_range = [0, edge]
            else:
                assert edge == edge_length, (
                    f'Length of {edge_name} must consist with '
                    f'{edge_range_name}, but length are {edge} and '
                    f'{edge_length} respectively.')

        return edge, edge_range

    def get_image_plane(self, H=None, W=None, H_range=None, W_range=None):
        """Get the size and  of the image plane.

        Args:
            H (int, optional)
        """
        H, H_range = self._handle_single_range(H, H_range, 'H')
        W, W_range = self._handle_single_range(W, W_range, 'W')
        return H, W, H_range, W_range

    def get_focal(self, fov=None, camera_angle_x=None):
        """Calculate the focal of the camera.
        Args:
            fov (float, optional): The 'Field of View' of the camera.
                Defaults to None.
            camera_angle_x (float, optional): Angle of the camera in x-axis.
                Defaults to None.

        Returns:
            float | None: Focal of the camera.
        """
        if (fov is None) and (camera_angle_x is None):
            return None

        if fov is not None:
            if self.degree2radian:
                fov = degree2radian(fov)
            focal_by_fov = 0.5 * self.H / np.tan(0.5 * fov)
        else:
            focal_by_fov = None

        if camera_angle_x is not None:
            if self.degree2radian:
                camera_angle_x = degree2radian(camera_angle_x)
            focal_by_cax = 0.5 * self.W / np.tan(0.5 * camera_angle_x)
        else:
            focal_by_cax = None

        if (focal_by_fov is not None) and (focal_by_cax is not None):
            assert focal_by_fov == focal_by_cax, 'conflict!'

        return focal_by_fov if focal_by_fov is not None else focal_by_cax

    def get_camera_matrix(self,
                          camera_matrix,
                          focal=None,
                          fov=None,
                          camera_angle_x=None):
        """Get the intrinsic matrix of the camera.

        Args:
            focal (float): The focal of the camera.

        returns:
            torch.Tensor: Intrinsic matrix in homo style, shape as [4, 4]
        """

        if camera_matrix is not None:
            focal = camera_matrix[0, 0]
            return camera_matrix, focal

        # calculate focal manually
        focal_ = self.get_focal(fov, camera_angle_x)
        assert (focal_ is not None) or (focal is not None), (
            'Can not calculate focal from given \'fov\' and '
            '\'camera_angle_x\', and \'focal\' is not passed either. Please '
            'pass at least one of \'fov\', \'camera_angle_x\' and \'focal\'.')

        # solve conflict
        if (focal is not None) and (focal_ is not None):
            assert focal == focal_, ''

        focal = focal if focal is not None else focal_
        camera_matrix = torch.eye(4)
        camera_matrix[0, 0] = focal
        camera_matrix[1, 1] = focal
        camera_matrix[0, -1] = sum(self.W_range) * 0.5
        camera_matrix[1, -1] = sum(self.H_range) * 0.5
        return camera_matrix, focal

    def get_random_pose(
            self,
            poses=None,
            num_batches=1,
            # homo=False,
            return_angle=False):
        """return camera position [x, y, z] in world coordinates if homo is
        True, return [x, y, z, 1]

        In this function, all sample operation is performed in radians.
        if self.degree2radians is True, theta_dist, phi_dist, radius_dist will
        be rescaled to radians

        theta: angle with x-axis
        phi: angle with x-y plane

        set y-axis as up

        return:
            torch.Tensor | tuple: [N, 3]
        """

        if poses is not None:
            return pose_to_tensor(poses)

        if self.camera_sample_mode == 'uniform':
            theta = uniform_sampling(num_batches, **self.theta_dist)
            phi = uniform_sampling(num_batches, **self.phi_dist)
        elif self.camera_sample_mode == 'gaussian':
            theta = gaussian_sampling(num_batches, **self.theta_dist)
            phi = gaussian_sampling(num_batches, **self.phi_dist)
        elif self.camera_sample_mode == 'spherical':
            u = uniform_sampling(num_batches, **self.u_dist)
            v = uniform_sampling(num_batches, **self.v_dist)
            theta = 2 * torch.from_numpy(np.pi) * u
            phi = torch.arccos(1 - 2 * v)
        else:
            raise ValueError()

        if (self.camera_sample_mode in ['uniform', 'gaussian']
                and self.degree2radian):
            theta = degree2radian(theta)
            phi = degree2radian(phi)
        r = uniform_sampling(num_batches, **self.radius_dist)

        camera_position = torch.zeros((num_batches, 3))
        camera_position[..., 0] = torch.cos(phi) * torch.cos(theta)
        camera_position[..., 1] = torch.sin(phi)
        camera_position[..., 2] = torch.cos(phi) * torch.sin(theta)
        camera_position = camera_position * r

        if return_angle:
            return camera_position, theta, phi
        return camera_position

    def get_camera2world(self, camera_position, at=None):
        """Get the camera2world, which is also the inverse (or transpose) of
        the 'look_at' matrix of the camera position.

        Args:
            camera_position: [[x, y, z]], shape as [bz, 3]
        RT = [[R], [T],
              [0], [1]]
        [bz, 4, 4]
        """
        if camera_position.ndim == 1:
            camera_position = camera_position.unsqueeze(0)

        batch_size = camera_position.shape[0]
        if at is None:
            # look at the origin of world coordinates as default
            at = torch.zeros(batch_size, 3)
        elif isinstance(at, np.ndarray):
            at = torch.from_numpy(at)
        elif isinstance(at, torch.Tensor):
            at = None
        up = self.up_axis.repeat([batch_size, 1])

        direction_vector = normalize_vector(camera_position - at)
        right_vector = normalize_vector(torch.cross(up, direction_vector))
        up_vector = normalize_vector(
            torch.cross(direction_vector, right_vector))

        RT = torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1)
        # stack at the last dimension, equals to transpose
        RT[:, :3, :3] = torch.stack(
            [right_vector, up_vector, direction_vector], dim=-1)
        RT[:, :3, 3] = camera_position
        return RT

    def plane_to_camera(self, plane, invert_y_axis=True, device=None):
        """Convert the image plane from `plane coordinates` to `camera
        coordinates`.

               |              ^ x
               |             /
        <------|      z <---+
               |            |
               |            V y
             plane

        Here we use [x, y, -1/f, 1] instead [x, y, -1, 1] to avoid matrix
        calculation
        because projection @ points =
        [[f 0 0 W],   [W,     [x,   [f(x-W),    [ (x-W)/f,
         [0 f 0 H],    H,      y,    f(y-H),      (y-H)/f,
         [0 0 1 0], @  0,   @ -f, =    -f,    =     -1,
         [0 0 0 1]]    1]      1]       1, ]        1/f  ]
         To be noted that, depth have no meaning, for we direct simplify
         [..., 1/f] to [..., -1]
        """
        plane[:, 0] = (plane[:, 0] - self.camera_matrix[0, -1]) / self.focal
        plane[:, 1] = (plane[:, 1] - self.camera_matrix[1, -1]) / self.focal
        plane[:, 2] = -1

        if invert_y_axis:
            plane[:, 1] = plane[:, 1] * -1
        if device is not None:
            plane = plane.to(device)
        return plane

    def sample_z_vals(self,
                      noise,
                      num_samples,
                      near=None,
                      far=None,
                      num_points=None,
                      device=None):
        """sample expanded z_vals in [near, far]
        Returns:
            [1, num_samples] or [n_points, n_samples]
        """

        # get necessasry sample parameters by default if not given
        near = self.near if near is None else near
        far = self.far if far is None else far

        assert near is not None
        assert far is not None

        z_vals = torch.linspace(near, far, num_samples).view(1, num_samples)
        if num_points is not None:
            z_vals = z_vals.expand(num_points, -1)

        if device is not None:
            z_vals = z_vals.to(device)

        if isinstance(noise, torch.Tensor):
            assert noise.shape == z_vals.shape
            z_vals = z_vals + noise
        elif callable(noise):
            z_vals = noise(z_vals)
        else:
            assert noise is None, (
                'Only support callable or torch.Tensor for noise, '
                f'but receive {type(noise)}')

        return z_vals

    def sample_rays(self,
                    camera_position,
                    plane=None,
                    plane_world=None,
                    camera2world=None):
        """
        Args:
            Camera position: Position of the current camera
            plane: plane in current coordinates
            plane_world: plane in the world coordinates
            camera2world: trans matrix from camera coordinates to the world
                coordinates
        Returns:
            dict(tensor): dict contains normalized view directions and rays
                vectors
        """
        camera_position = prepare_vector(
            camera_position, to_matrix=False, is_batch=False)

        if plane is not None:
            # shape as [H*W, 4]
            rays = prepare_vector(plane, to_matrix=True)
            # transform rays to the world coordinates. To be noted that
            # `plane` is a group of *points* this means the last variable of
            # the coordinates is non-zero, but `rays` is a group of *vectors*,
            # Therefore we directly convert the last variable to 0.
            rays[..., -1, :] = 0
            camera2world = prepare_matrix(camera2world)
            rays = (camera2world @ rays).squeeze()

        elif plane_world is not None:
            # shape as [H*W, 4]
            plane_world = prepare_vector(plane_world, to_matrix=False)
            # [H*W, 4] = [H*W, 4] - [4, ]
            rays = plane_world - camera_position
        else:
            # TODO: revise this message
            raise ValueError(
                'Cannot calculate rays from current input. Please make sure '
                'that plane is not None or plane_world is not None.')

        views = normalize_vector(rays)
        return dict(views=views, rays=rays)

    def sample_render_points(self,
                             num_samples,
                             camera_position,
                             plane=None,
                             plane_world=None,
                             near=None,
                             far=None,
                             camera2world=None,
                             noise=None):
        """plane_world and plane can be transfor to each other, we mainlly
        calculate rays by those two input. we have.

        rays = plane = plane_world - camera_position, if plane is not None,
        plane will be preferentially
        Args:
            num_samples: number of samples on each ray.
            camera_position: [4]
            plane : [H*W, 4] for non homo and [H*W, 4] for homo, plane points
                in camera coordinates.

            plane_world: [H*W, 4], plane coordinates in world coordinates
            noise (torch.Tensor, fn):

        Returns:
            points:
            z_vals:
        """
        # shape as [4, ]
        camera_position = prepare_vector(
            camera_position, to_matrix=False, is_batch=False)

        if plane is not None:
            # shape as [H*W, 4]
            rays = prepare_vector(plane, to_matrix=True)
            # transform rays to the world coordinates. To be noted that
            # `plane` is a group of *points* this means the last variable of
            # the coordinates is non-zero, but `rays` is a group of *vectors*,
            # Therefore we directly convert the last variable to 0.
            rays[..., -1, :] = 0
            camera2world = prepare_matrix(camera2world)
            rays = (camera2world @ rays).squeeze()

        elif plane_world is not None:
            # shape as [H*W, 4]
            plane_world = prepare_vector(plane_world, to_matrix=False)
            # [H*W, 4] = [H*W, 4] - [4, ]
            rays = plane_world - camera_position
        else:
            # TODO: revise this message
            raise ValueError(
                'Cannot calculate rays from current input. Please make sure '
                'that plane is not None or plane_world is not None.')

        views = normalize_vector(rays)

        # [H*W, N, 1]
        z_vals = self.sample_z_vals(
            noise, num_samples, near=near, far=far,
            num_points=rays.shape[0]).to(views.device)

        # expand at the -2 dim in order to broadcast with N_samples
        # [H*W, N, 4] = [H*W, 1, 4] * [H*W, N, 1] + [4]
        points = rays[:, None, :] * z_vals[..., None] + camera_position

        output_dict = dict(
            views=views, rays=rays, points=points, z_vals=z_vals)
        return output_dict

    def points_to_world(self, points, camera2world=None):
        """
        points: [N, 3] or [N, 4]
        camera2world: [4, 4] or [3, 3]
        points: [N, 4] points to render
        """
        # reshape camera2world as [bz, 4, 4]
        camera2world = prepare_matrix(camera2world)
        points = prepare_vector(points)

        N = points.shape[0]
        points_in_world = camera2world.expand(N, -1, -1) @ \
            points.view(-1, 4, 1)
        return points_in_world.view(N, -1)

    def split_transform_matrix(self, transform_matrix):
        """
        This function is only used for nerf dataset.
        Args:
            transform_matrix: [4, 4]

        Returns:
            camera2world: [4, 4]
            pose: [4, ]
        """
        # add shape check and batch level handling
        camera2world = transform_matrix.clone()
        pose = torch.ones((4, ))
        pose[:3] = transform_matrix[:3, -1]
        return camera2world.float(), pose.float()

    def select_pixels(self, imgs, coords=None):
        """
        Args:
            imgs: [3, H, W]
            selected_idx: [num_poinst, ]

        Returns:
            [num_points, 3]
        """
        imgs_ = imgs.clone().permute(1, 2, 0)
        imgs_ = imgs_[coords[0], coords[1], :]

        return imgs_

    def get_camera_info(self,
                        camera_pose=None,
                        camera2world=None,
                        transform_matrix=None,
                        device=None):
        # prepare for camera2world matrix and pose
        if transform_matrix is not None:
            # split camera2world and poses
            camera2world, camera_pose = self.split_transform_matrix(
                transform_matrix)
        else:
            camera_pose = self.get_random_pose(camera_pose)
            camera2world = self.get_camera2world(camera_pose)

        output_dict = dict(camera_pose=camera_pose, camera2world=camera2world)

        if device is not None:
            output_dict = {k: v.to(device) for k, v in output_dict.items()}
        return output_dict

    def sample_points(self, rays, z_vals, camera_position):
        """
        Args:
            rays: [n_points, 4]
            z_vals: [1/n_points, n_samples]
            camera_position: [4]
        """

        # [n_points', n_samples, 4] =
        #   [n_points', 1, 4] * [1/n_points', n_samples, 1] + [4,]
        points = rays[:, None, :] * z_vals[..., None] + camera_position
        return points

    def prepare_render_rays(self,
                            real_img=None,
                            plane_world=None,
                            camera_pose=None,
                            transform_matrix=None,
                            device=None,
                            **kwargs):

        # sample points
        sample_dict = self.ray_sampler.sample_rays(real_img)
        sample_dict['points_selected'] = self.plane_to_camera(
            sample_dict['points_selected'], invert_y_axis=True, device=device)

        # prepare for camera2world matrix and pose
        if transform_matrix is not None:
            # split camera2world and poses
            camera2world, camera_pose = self.split_transform_matrix(
                transform_matrix)
            camera2world = camera2world.to(device)
            camera_pose = camera_pose.to(device)
        else:
            camera_pose = self.get_random_pose(camera_pose).to(device)
            camera2world = self.get_camera2world(camera_pose).to(device)
        sample_dict['camera_pose'] = camera_pose

        # actually return a dict contains ray-vector and view vectors
        # maybe we should change name
        ray_dict = self.sample_rays(camera_pose,
                                    sample_dict['points_selected'],
                                    plane_world, camera2world)

        sample_dict.update(ray_dict)
        # return output_dict
        return sample_dict
