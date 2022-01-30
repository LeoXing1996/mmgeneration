from copy import deepcopy

import torch

from mmgen.models import build_module
from mmgen.models.builder import MODULES


class TestPoseCamera:

    @classmethod
    def setup_class(cls):
        cls.n_points = 3
        cls.default_ray_sampler = dict(
            type='FullRaySampler', n_points=cls.n_points)
        cls.default_cfg = dict(
            type='PoseCamera',
            fov=30,
            H_range=[0, 4],
            W_range=[0, 4],
            near=7.5,
            far=12.5,
            which_hand='left',
            ray_sampler=cls.default_ray_sampler,
            degree2radian=True)
        cls.real_img = torch.randn(2, 3, 4, 4)

    def test_pose_camera(self):
        camera = build_module(self.default_cfg)
        assert camera.is_homo and camera.num_coors == 4
        # assert camera.n_points == 1
        assert (camera.up_axis == torch.FloatTensor([0, 0, 1])).all()
        assert camera.near == 7.5
        assert camera.far == 12.5

        # test prepare_render_rays --> input transform_matrix
        transform_matrix = torch.arange(0, 2 * 4 * 4).reshape([2, 4, 4])
        render_dict = camera.prepare_render_rays(
            batch_size=2, transform_matrix=transform_matrix)

        # shape checking
        assert all(
            [v.shape[:2] == (2, self.n_points) for v in render_dict.values()])

        # ensure camera pose is same for all points in each sample
        for idx in range(2):
            poses = render_dict['camera_pose'][idx]
            for pose in poses:
                assert (pose == poses[0]).all()

        # test use default ray sampler
        config = deepcopy(self.default_cfg)
        config['ray_sampler'] = None
        camera = build_module(config)
        assert isinstance(camera.ray_sampler, MODULES['FullRaySampler'])


class TestRandomPoseCamera:

    @classmethod
    def setup_class(cls):
        cls.n_points = 1
        cls.default_ray_sampler = dict(
            type='FullRaySampler', n_points=cls.n_points)
        cls.default_cfg = dict(
            type='RandomPoseCamera',
            fov=30,
            H_range=[0, 4],
            W_range=[0, 4],
            near=7.5,
            far=12.5,
            which_hand='left',
            camera_sample_mode='spherical',
            u_dist=[0, 1],
            v_dist=[0, 0.45642212862617093],
            radius_dist=10,
            ray_sampler=cls.default_ray_sampler,
            degree2radian=True)
        cls.real_img = torch.randn(2, 3, 4, 4)

    def test_random_pose_camera(self):
        camera = build_module(self.default_cfg)

        render_dict = camera.prepare_render_rays(real_img=self.real_img)
        # shape checking
        assert all(
            [v.shape[:2] == (2, self.n_points) for v in render_dict.values()])
