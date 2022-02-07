from copy import deepcopy

import numpy as np
import torch

from mmgen.models import build_module


class TestFullRaySampler:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type='FullRaySampler', H_range=[0, 10], W_range=[0, 10])
        cls.batch_size = 2
        cls.image = torch.rand(2, 3, 10, 10)

    def test_full_plane(self):
        ray_sampler = build_module(self.default_cfg)

        # test self.train and self.eval
        assert ray_sampler.training is True
        ray_sampler.eval()
        assert not ray_sampler.training
        ray_sampler.train()
        assert ray_sampler.training
        ray_sampler.train(False)
        assert not ray_sampler.training

        # test sample plane
        plane = ray_sampler.sample_plane_full(batch_size=1)
        assert plane.shape == (1, 100, 4)

        # test sample_ray
        sample_dict = ray_sampler.sample_rays(self.batch_size, self.image)
        assert all([
            key in sample_dict
            for key in ['real_pixels', 'selected_idx', 'points_selected']
        ])
        assert sample_dict['real_pixels'].shape == (2, 100, 3)
        # print('real_pixels: ', type(sample_dict['real_pixels']))
        assert (sample_dict['real_pixels'] == self.image.permute(
            0, 2, 3, 1).reshape(2, 100, 3)).all()
        assert sample_dict['selected_idx'].shape == (2, 100)
        # print('selected_idx: ', type(sample_dict['selected_idx']))
        assert (sample_dict['selected_idx'] == torch.arange(0, 100)).all()
        assert sample_dict['points_selected'].shape == (2, 100, 4)
        assert (sample_dict['points_selected'] == plane).all()
        # print('points_selected: ', type(sample_dict['points_selected']))

        # test sample_ray without image + is_homo is False
        config = deepcopy(self.default_cfg)
        config['is_homo'] = False
        ray_sampler = build_module(config)
        sample_dict = ray_sampler.sample_rays(1)
        assert sample_dict['selected_idx'].shape == (1, 100)
        assert (sample_dict['selected_idx'] == torch.arange(0, 100)).all()
        assert sample_dict['points_selected'].shape == (1, 100, 3)
        assert (sample_dict['points_selected'] == plane[..., :3]).all()
        assert 'real_pixels' not in sample_dict


class TestRaySampler:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type='RaySampler', H_range=[0, 10], W_range=[0, 10], n_points=10)
        cls.batch_size = 2
        cls.image = torch.rand(2, 3, 10, 10)

    def test_ray_sampler(self):
        # --- test under train mode ---
        ray_sampler = build_module(self.default_cfg)
        assert ray_sampler.training is True

        # get a full plane
        plane = ray_sampler.sample_plane_full(batch_size=1)

        # test sample_rays
        sample_dict = ray_sampler.sample_rays(self.batch_size, self.image)
        assert all([
            key in sample_dict
            for key in ['real_pixels', 'selected_idx', 'points_selected']
        ])
        assert sample_dict['real_pixels'].shape == (2, 10, 3)
        assert sample_dict['selected_idx'].shape == (2, 10)
        assert sample_dict['points_selected'].shape == (2, 10, 4)
        # print('selected_idx: ', type(sample_dict['selected_idx']))
        # print('points_selected: ', type(sample_dict['points_selected']))

        # try to do this points_selection manually
        points_selected_np = np.take_along_axis(
            plane.repeat(2, 1, 1).numpy(),
            sample_dict['selected_idx'].numpy()[..., None],
            axis=1)
        assert (points_selected_np == sample_dict['points_selected'].numpy()
                ).all()

        # --- test under eval mode (sample all points) ---
        ray_sampler.eval()
        assert not ray_sampler.training
        sample_dict = ray_sampler.sample_rays(self.batch_size, self.image)
        assert all([
            key in sample_dict
            for key in ['real_pixels', 'selected_idx', 'points_selected']
        ])
        assert sample_dict['real_pixels'].shape == (2, 100, 3)
        assert (sample_dict['real_pixels'] == self.image.permute(
            0, 2, 3, 1).reshape(2, 100, 3)).all()
        assert sample_dict['selected_idx'].shape == (2, 100)
        assert (sample_dict['selected_idx'] == torch.arange(0, 100)).all()
        assert sample_dict['points_selected'].shape == (2, 100, 4)
        assert (sample_dict['points_selected'] == plane).all()


class TestPrecropSampler:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type='PrecropRaySampler',
            H_range=[0, 10],
            W_range=[0, 10],
            precrop_frac=0.5,
            n_points=10)
        cls.batch_size = 2
        cls.image = torch.rand(2, 3, 10, 10)

    def test_precrop_raysampler(self):
        # --- test under train mode ---
        ray_sampler = build_module(self.default_cfg)
        assert ray_sampler.training is True

        plane = ray_sampler.sample_plane_full(batch_size=2)

        # test select index
        selected_index = ray_sampler.select_index(plane)
        assert selected_index.shape == (2, 10)

        # test sample_rays
        sample_dict = ray_sampler.sample_rays(self.batch_size, self.image)
        assert all([
            key in sample_dict
            for key in ['real_pixels', 'selected_idx', 'points_selected']
        ])
        assert sample_dict['real_pixels'].shape == (2, 10, 3)
        assert sample_dict['selected_idx'].shape == (2, 10)
        assert sample_dict['points_selected'].shape == (2, 10, 4)

        # check range
        assert (3 <= sample_dict['points_selected'][..., 0]).all()
        assert (3 <= sample_dict['points_selected'][..., 1]).all()
        assert (sample_dict['points_selected'][..., 0] <= 6).all()
        assert (sample_dict['points_selected'][..., 1] <= 6).all()

        # --- test under eval mode + set precrop frac---
        ray_sampler.eval()
        assert not ray_sampler.training

        # test set precrop_frac --> select [1, 8]
        ray_sampler.set_precrop_frac(0.8)
        assert ray_sampler.curr_precrop_frac == 0.8

        # test select index
        selected_index = ray_sampler.select_index(plane)
        assert selected_index.shape == (2, 64)

        # test sample_rays
        sample_dict = ray_sampler.sample_rays(self.batch_size, self.image)
        assert all([
            key in sample_dict
            for key in ['real_pixels', 'selected_idx', 'points_selected']
        ])
        assert sample_dict['real_pixels'].shape == (2, 64, 3)
        assert sample_dict['selected_idx'].shape == (2, 64)
        assert sample_dict['points_selected'].shape == (2, 64, 4)

        # check range
        x, y = torch.meshgrid(torch.arange(1, 9), torch.arange(1, 9))
        assert (y.reshape(1, -1) == sample_dict['points_selected'][...,
                                                                   0]).all()
        assert (x.reshape(1, -1) == sample_dict['points_selected'][...,
                                                                   1]).all()


class TestFlexGridRaySampler:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type='FlexGridRaySampler',
            H_range=[0, 10],
            W_range=[0, 10],
            min_scale=0.25,
            max_scale=1,
            random_shift=True,
            random_scale=True,
            n_points=9)
        cls.batch_size = 2
        cls.image = torch.rand(2, 3, 10, 10)

    def test_flexgrid_raysampler(self):
        # --- test under train mode ---
        ray_sampler = build_module(self.default_cfg)
        assert ray_sampler.training
        assert ray_sampler.curr_min_scale == 0.25

        plane = ray_sampler.sample_plane_full(batch_size=2)

        # test select index
        selected_coord = ray_sampler.select_index(plane)
        assert selected_coord.shape == (2, 3, 3, 2)

        # test sample_rays
        sample_dict = ray_sampler.sample_rays(self.batch_size, self.image)
        assert all([
            key in sample_dict
            for key in ['real_pixels', 'selected_idx', 'points_selected']
        ])
        assert sample_dict['selected_idx'].shape == (2, 3, 3, 2)
        assert sample_dict['real_pixels'].shape == (2, 9, 3)
        assert sample_dict['points_selected'].shape == (2, 9, 4)

        # TODO: maybe we can add some code to test the sample results

        # test set scale
        ray_sampler.set_min_scale(0.3)
        assert ray_sampler.curr_min_scale == 0.3

        # --- test under test mode ---
        ray_sampler.eval()
        assert not ray_sampler.training
        sample_dict = ray_sampler.sample_rays(self.batch_size, self.image)
        assert all([
            key in sample_dict
            for key in ['real_pixels', 'selected_idx', 'points_selected']
        ])
        #  NOTE: 'selected_idx' have no meaning in eval mode, not test this in
        # unit test
        assert sample_dict['real_pixels'].shape == (2, 100, 3)
        assert sample_dict['points_selected'].shape == (2, 100, 4)

        # test random_scale is False + random shift is False
        config = deepcopy(self.default_cfg)
        config['random_shift'] = False
        config['random_scale'] = False
        ray_sampler = build_module(config)
        assert ray_sampler.training

        plane = ray_sampler.sample_plane_full(batch_size=2)

        # test select index
        selected_coord = ray_sampler.select_index(plane)
        assert selected_coord.shape == (2, 3, 3, 2)
        assert (selected_coord == ray_sampler._get_base_coord()[None,
                                                                ...]).all()
