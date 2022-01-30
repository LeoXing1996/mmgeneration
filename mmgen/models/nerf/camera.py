# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import torch
from mmcv.utils import is_list_of

from mmgen.models.builder import MODULES, build_module
from .util import (degree2radian, gaussian_sampling, normalize_vector,
                   pose_to_tensor, prepare_vector, uniform_sampling)


# TODO: maybe change another name, Pose means the pose is given
@MODULES.register_module('PoseCamera')
@MODULES.register_module()
class Camera(object):
    r"""
    camera_sample_dist is a dict contains the distribution to sample a random
    camera.

    In this class, we all use homogenerous coordinates

    .. code-block:: python
        :linenons:

    Args:
        camera_sample_dist (dict):


    camera_angle_x: angle on x-axis.
    """

    _default_ray_sampler = dict(type='FullRaySampler', n_points=None)

    def __init__(self,
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
                 which_hand='right',
                 degree2radian=False,
                 is_homo=True):

        self.H, self.W, self.H_range, self.W_range = self.get_image_plane(
            H, W, H_range, W_range)

        self.is_homo = is_homo

        # build ray sampler
        if ray_sampler is None:
            ray_sampler = self._default_ray_sampler
        ray_sampler_ = deepcopy(ray_sampler)
        ray_sampler_['H_range'] = self.H_range
        ray_sampler_['W_range'] = self.W_range
        ray_sampler_['is_homo'] = self.is_homo
        self.ray_sampler = build_module(ray_sampler_)

        self.near, self.far = near, far
        self.degree2radian = degree2radian

        # init instrinst matrix / projection matrix / camera matrix
        self.camera_matrix, self.focal = self.get_camera_matrix(
            camera_matrix, focal, fov, camera_angle_x)

        assert which_hand in ['right', 'left']
        self.which_hand = which_hand
        if self.which_hand == 'right':
            self.up_axis = torch.FloatTensor([0, 1, 0]).unsqueeze(0)
        else:
            self.up_axis = torch.FloatTensor([0, 0, 1]).unsqueeze(0)

    @property
    def num_coors(self):
        """The number of coordinates."""
        return 4 if self.is_homo else 3

    @property
    def n_points(self):
        return self.ray_sampler.n_points

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

        # use repeat to allocate new memory
        RT = torch.eye(4)[None, ].repeat(batch_size, 1, 1)
        # stack at the last dimension, equals to transpose
        RT[:, :3, :3] = torch.stack(
            [right_vector, up_vector, direction_vector], dim=-1)
        RT[:, :3, 3] = camera_position
        return RT

    def plane_to_camera(self, plane, invert_y_axis=True):
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

        Args:
            plane (torch.Tensor): Coordinates of the plane to render. Shape
                like ``[batch_size, n_points, 3/4]``.

        Returns:
            torch.Tensor: Coordinates converted to the camera coordinates.
        """
        plane[...,
              0] = (plane[..., 0] - self.camera_matrix[0, -1]) / self.focal
        plane[...,
              1] = (plane[..., 1] - self.camera_matrix[1, -1]) / self.focal
        plane[..., 2] = -1

        if invert_y_axis:
            plane[..., 1] = plane[..., 1] * -1

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

    def rays_to_world(self,
                      camera_position,
                      plane=None,
                      plane_world=None,
                      camera2world=None):
        """
        Args:
            Camera position: Position of the current camera [bz*n_points, 3/4]
            plane: plane in current coordinates [bz, n_points, 3/4]
            plane_world: plane in the world coordinates [bz, n_points, 3/4]
            camera2world: trans matrix from camera coordinates to the world
                coordinates [bz, 3/4]
        Returns:
            dict(tensor): dict contains normalized view directions and rays
                vectors. All shape as ``[n_points, n_samples]``
        """
        # [bz, 3] -> [bz, 4] (to homo here)
        # camera_position = prepare_vector(camera_position, to_matrix=False)

        if plane is not None:
            # [bz, n_points, 4] -> [bz, n_points, 4, 1] for bmm
            rays = plane[..., None]
            # transform rays to the world coordinates. To be noted that
            # `plane` is a group of *points* this means the last variable of
            # the coordinates is non-zero, but `rays` is a group of *vectors*,
            # Therefore we directly convert the last variable to 0.
            rays[..., -1, :] = 0
            rays = rays.reshape([-1, 4, 1])
            # [bz*n_points, 4, 4] @ [bz*n_points, 4, 1] = [bz*n_points, 4, 1]
            rays = (camera2world @ rays).squeeze()

        elif plane_world is not None:
            # TODO: support batch
            # shape as [H*W, 4]
            plane_world = prepare_vector(plane_world, to_matrix=False)
            # [H*W, 4] = [H*W, 4] - [4, ]
            rays = plane_world - camera_position
        else:
            raise ValueError(
                'Cannot calculate rays from current input. Please make sure '
                'that \'plane\' and \'plane_world\' are not both None.')
        views = normalize_vector(rays)

        return dict(views=views, rays=rays)

    def split_transform_matrix(self, transform_matrix):
        """Split camera2world and poses. This function is only used for nerf
        dataset.

        Args:
            transform_matrix: [batch_size, 4, 4]

        Returns:
            camera2world: [batch_size, 4, 4]
            pose: [batch_size, 4]
        """
        # add shape check and batch level handling
        batch_size = transform_matrix.shape[0]
        camera2world = transform_matrix.clone()
        pose = torch.ones(batch_size, self.num_coors)
        pose[:, :3] = transform_matrix[:, :3, -1]
        return camera2world.float(), pose.float()

    def get_camera_info(self, transform_matrix=None):
        """
        Returns:
            camera_pose: [bz*n_points, 3/4]
            camera2world: [bz*n_points, 3/4]
        """
        camera2world, camera_pose = self.split_transform_matrix(
            transform_matrix)

        # repeat each batch size and flatten
        n_points = self.ray_sampler.n_points
        n_coors = self.num_coors
        camera2world = camera2world[:, None].repeat([1, n_points, 1, 1])
        camera_pose = camera_pose[:, None].repeat([1, n_points, 1])
        camera2world = camera2world.reshape([-1, n_coors, n_coors])
        camera_pose = camera_pose.reshape([-1, n_coors])

        return camera_pose, camera2world

    def sample_points(self, rays, z_vals, camera_position):
        """
        Args:
            rays: [n_points, 4]
            z_vals: [1/n_points, n_samples]
            camera_position: [4]
        """
        # [n_points', n_samples, 4] =
        #   [n_points', 1, 4] * [n_points', n_samples, 1] + [n_points', 1, 4]
        points = rays[:, None, :] * z_vals[..., None] + \
            camera_position[:, None]
        return points

    def prepare_render_rays(self,
                            batch_size=1,
                            real_img=None,
                            plane_world=None,
                            transform_matrix=None,
                            device=None):
        """Return the sampled points and pixels with the given pose."""

        # sample points
        sample_dict = self.ray_sampler.sample_rays(
            batch_size=batch_size, image=real_img)
        sample_dict['points_selected'] = self.plane_to_camera(
            sample_dict['points_selected'], invert_y_axis=True)

        # prepare for camera2world matrix and pose
        camera_pose, camera2world = self.get_camera_info(transform_matrix)
        sample_dict['camera_pose'] = camera_pose

        ray_dict = self.rays_to_world(camera_pose,
                                      sample_dict['points_selected'],
                                      plane_world, camera2world)

        sample_dict.update(ray_dict)

        target_shape = (batch_size, self.ray_sampler.n_points, -1)
        for k, v in sample_dict.items():
            v = v.reshape(target_shape)
            if device is not None:
                v = v.to(device)
            sample_dict[k] = v

        return sample_dict


@MODULES.register_module()
class RandomPoseCamera(Camera):
    """Support sample random pose automatically.

    ``sample_mode = 'spherical'``, the dict should be like:

    .. code-block:: python
        :linenons:
    """

    _support_sample_dist = [
        'uniform', 'gaussian', 'normal', 'truncated_gaussian', 'spherical',
        'grid'
    ]

    def __init__(
            self,
            u_dist=None,
            v_dist=None,
            theta_dist=None,
            phi_dist=None,
            radius_dist=1,
            camera_sample_mode=None,
            is_hemi_sphere=False,  # TODO: clip the sample range
            *args,
            **kwargs):

        super().__init__(*args, **kwargs)

        # convert input to dict and setattr
        dist_list = [u_dist, v_dist, theta_dist, phi_dist, radius_dist]
        dist_name_list = ['u', 'v', 'theta', 'phi', 'radius']
        for name, conf in zip(dist_name_list, dist_list):
            if isinstance(conf, dict):
                setattr(self, f'{name}_dist', conf)
            elif isinstance(conf, int):
                setattr(self, f'{name}_dist', dict(val=conf))
            elif isinstance(conf, list):
                assert all([
                    isinstance(val, (float, int)) for val in conf
                ]) and len(conf) == 2, (
                    'The input distribution with list type must be list of '
                    'int or float and the length of the list must be 2. But '
                    f'receive \'{conf}\'.')
                setattr(self, f'{name}_dist',
                        dict(lower_bound=min(conf), upper_bound=max(conf)))
            else:
                assert conf is None, (
                    f'Camera sample distribution of \'{name}\' should be '
                    f'None, dict or val. But receive {conf}.')

        assert camera_sample_mode in self._support_sample_dist
        self.camera_sample_mode = camera_sample_mode

    def get_random_pose(self, poses=None, num_batches=1, return_angle=False):
        """return camera position [x, y, z] in world coordinates if homo is
        True, return [x, y, z, 1]

        In this function, all sample operation is performed in radians.
        if self.degree2radians is True, theta_dist, phi_dist, radius_dist will
        be rescaled to radians

        theta: angle with x-axis
        phi: angle with z-axis

        set y-axis as up

        return:
            torch.Tensor | tuple: [N, 3/4]
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
            # NOTE: just for debug
            np.random.seed(0)

            u = uniform_sampling(num_batches, **self.u_dist)
            v = uniform_sampling(num_batches, **self.v_dist)
            theta = 2 * np.pi * u
            phi = torch.arccos(1 - 2 * v)
        else:
            raise ValueError()

        if (self.camera_sample_mode in ['uniform', 'gaussian']
                and self.degree2radian):
            theta = degree2radian(theta)
            phi = degree2radian(phi)

        r = uniform_sampling(num_batches, **self.radius_dist)

        # TODO: here we set [0, 0, 1] as the up-axis,
        # and this is the left-hand coordinates, we should support
        # coordinates
        camera_position = torch.zeros((num_batches, 3))
        camera_position[..., 0] = torch.sin(phi) * torch.cos(theta)
        camera_position[..., 1] = torch.sin(phi) * torch.sin(theta)
        camera_position[..., 2] = torch.cos(phi)

        camera_position = camera_position * r

        if return_angle:
            return camera_position, theta, phi
        return camera_position

    def get_camera_info(self, camera_pose=None, num_batches=1):
        """Get the camera info. If ``camera_pose`` is not passed, we will try
        to sample a random pose.

        Returns:
            camera_pose: [bz*n_points, 3/4]
            camera2world: [bz*n_points, 3/4, 3/4]
        """
        # TODO: here we should input a batch size but not camera pose
        camera_pose = self.get_random_pose(
            camera_pose, num_batches=num_batches)
        camera2world = self.get_camera2world(camera_pose)

        if self.is_homo:
            # fill 1 at end
            camera_pose = torch.cat(
                [camera_pose, torch.ones(num_batches, 1)], dim=-1)

        # repeat for each batch size and flatten
        n_points = self.ray_sampler.n_points
        n_coors = self.num_coors
        camera2world = camera2world[:, None].repeat([1, n_points, 1, 1])
        camera_pose = camera_pose[:, None].repeat([1, n_points, 1])
        camera2world = camera2world.reshape([-1, n_coors, n_coors])
        camera_pose = camera_pose.reshape([-1, n_coors])

        return camera_pose, camera2world

    def prepare_render_rays(self,
                            batch_size=1,
                            real_img=None,
                            plane_world=None,
                            camera_pose=None,
                            device=None):
        """Return the sampled points and pixels with the given pose.

        Returns:
            dict: A dict contains parameters related to camera and render.
                To be noted that, in current version variables related to
                render (``rays`` and ``views``) are flattened to `[bz*H*W, N]`.
        """

        # sample points
        sample_dict = self.ray_sampler.sample_rays(
            batch_size=batch_size, image=real_img)
        sample_dict['points_selected'] = self.plane_to_camera(
            sample_dict['points_selected'], invert_y_axis=True)

        # get batch size from sample dict, because we may use batch size from
        # real image
        if real_img is not None:
            batch_size = real_img.shape[0]

        # prepare for camera2world matrix and pose
        camera_pose, camera2world = self.get_camera_info(
            camera_pose, num_batches=batch_size)
        sample_dict['camera_pose'] = camera_pose

        # actually return a dict contains ray-vector and view vectors
        # maybe we should change name
        ray_dict = self.rays_to_world(camera_pose,
                                      sample_dict['points_selected'],
                                      plane_world, camera2world)

        sample_dict.update(ray_dict)

        # reshape to [bz, H*W, N]
        target_shape = (batch_size, self.ray_sampler.n_points, -1)
        for k, v in sample_dict.items():
            v = v.reshape(target_shape)
            if device is not None:
                v = v.to(device)
            sample_dict[k] = v

        # if device is not None:
        #     sample_dict = {k: v.to(device) for k, v in sample_dict.items()}

        return sample_dict
