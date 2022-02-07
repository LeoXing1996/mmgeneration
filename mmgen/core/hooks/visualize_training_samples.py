# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from copy import deepcopy

import mmcv
import numpy as np
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only
from torchvision.utils import save_image


@HOOKS.register_module()
class VisualizeUnconditionalSamples(Hook):
    """Visualization hook for unconditional GANs.

    In this hook, we use the official api `save_image` in torchvision to save
    the visualization results.

    Args:
        output_dir (str): The file path to store visualizations.
        fixed_noise (bool, optional): Whether to use fixed noises in sampling.
            Defaults to True.
        num_samples (int, optional): The number of samples to show in
            visualization. Defaults to 16.
        interval (int): The interval of calling this hook. If set to -1,
            the visualization hook will not be called. Default: -1.
        filename_tmpl (str): Format string used to save images. The output file
            name will be formatted as this args. Default: 'iter_{}.png'.
        rerange (bool): Whether to rerange the output value from [-1, 1] to
            [0, 1]. We highly recommend users should preprocess the
            visualization results on their own. Here, we just provide a simple
            interface. Default: True.
        bgr2rgb (bool): Whether to reformat the channel dimension from BGR to
            RGB. The final image we will save is following RGB style.
            Default: True.
        nrow (int): The number of samples in a row. Default: 1.
        padding (int): The number of padding pixels between each samples.
            Default: 4.
        kwargs (dict | None, optional): Key-word arguments for sampling
            function. Defaults to None.
    """

    def __init__(self,
                 output_dir,
                 fixed_noise=True,
                 num_samples=16,
                 interval=-1,
                 filename_tmpl='iter_{}.png',
                 rerange=True,
                 bgr2rgb=True,
                 nrow=4,
                 padding=0,
                 kwargs=None):
        self.output_dir = output_dir
        self.fixed_noise = fixed_noise
        self.num_samples = num_samples
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        self.bgr2rgb = bgr2rgb
        self.rerange = rerange
        self.nrow = nrow
        self.padding = padding

        # the sampling noise will be initialized by the first sampling.
        self.sampling_noise = None

        self.kwargs = kwargs if kwargs is not None else dict()

    @master_only
    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        # eval mode
        runner.model.eval()
        # no grad in sampling
        with torch.no_grad():
            outputs_dict = runner.model(
                self.sampling_noise,
                return_loss=False,
                num_batches=self.num_samples,
                return_noise=True,
                **self.kwargs)
            imgs = outputs_dict['fake_img']
            noise_ = outputs_dict['noise_batch']
        # initialize samling noise with the first returned noise
        if self.sampling_noise is None and self.fixed_noise:
            self.sampling_noise = noise_

        # train mode
        runner.model.train()

        filename = self.filename_tmpl.format(runner.iter + 1)
        if self.rerange:
            imgs = ((imgs + 1) / 2)
        if self.bgr2rgb and imgs.size(1) == 3:
            imgs = imgs[:, [2, 1, 0], ...]
        if imgs.size(1) == 1:
            imgs = torch.cat([imgs, imgs, imgs], dim=1)
        imgs = imgs.clamp_(0, 1)

        mmcv.mkdir_or_exist(osp.join(runner.work_dir, self.output_dir))
        save_image(
            imgs,
            osp.join(runner.work_dir, self.output_dir, filename),
            nrow=self.nrow,
            padding=self.padding)


# TODO: maybe rename to visNerf...
@HOOKS.register_module()
class VisualizeNeRFSamples(Hook):
    """Visualization hook for NeRF models.

    In this hook, we use the official api `save_image` in torchvision to save
    the visualization results.

    Args:
        output_dir (str): The file path to store visualizations.
        fixed_noise (bool, optional): Whether to use fixed noises in sampling.
            Defaults to True.
        num_samples (int, optional): The number of samples to show in
            visualization. Defaults to 16.
        interval (int): The interval of calling this hook. If set to -1,
            the visualization hook will not be called. Default: -1.
        filename_tmpl (str): Format string used to save images. The output file
            name will be formatted as this args. Default: 'iter_{}.png'.
        rerange (bool): Whether to rerange the output value from [-1, 1] to
            [0, 1]. We highly recommend users should preprocess the
            visualization results on their own. Here, we just provide a simple
            interface. Default: True.
        bgr2rgb (bool): Whether to reformat the channel dimension from BGR to
            RGB. The final image we will save is following RGB style.
            Default: True.
        nrow (int): The number of samples in a row. Default: 1.
        padding (int): The number of padding pixels between each samples.
            Default: 4.
        kwargs (dict | None, optional): Key-word arguments for sampling
            function. Defaults to None.
    """

    # TODO: support this in args
    _color_map = [
        [0, 0, 0],  # 0.000
        [0, 0, 255],  # 0.114
        [255, 0, 0],  # 0.299
        [255, 0, 255],  # 0.413
        [0, 255, 0],  # 0.587
        [0, 255, 255],  # 0.701
        [255, 255, 0],  # 0.886
        [255, 255, 255],  # 1.000
        [255, 255, 255],  # 1.000
    ]

    _color_map_bincenters = [
        0.0,
        0.114,
        0.299,
        0.413,
        0.587,
        0.701,
        0.886,
        1.000,
        2.000,  # doesn't make a difference, just strictly higher than 1
    ]

    def __init__(self,
                 output_dir,
                 dataloader=None,
                 dist=True,
                 num_samples=16,
                 interval=-1,
                 filename_tmpl='iter_{}.png',
                 rerange=True,
                 bgr2rgb=True,
                 nrow=4,
                 padding=0,
                 vis_keys=None,
                 kwargs=None):
        self.output_dir = output_dir
        self.dataloader = dataloader
        self.dist = dist
        self.num_samples = num_samples if num_samples != -1 \
            else len(dataloader.dataset)
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        self.bgr2rgb = bgr2rgb
        self.rerange = rerange
        self.nrow = nrow
        self.padding = padding

        self.vis_keys = vis_keys if vis_keys is not None else 'fake_img'
        if not isinstance(self.vis_keys, list):
            self.vis_keys = [self.vis_keys]
        self.kwargs = kwargs if kwargs is not None else dict()

    def _save_image(self, output_dict, runner):

        mmcv.mkdir_or_exist(osp.join(runner.work_dir, self.output_dir))

        if len(self.vis_keys) == 1:
            filenames = [self.filename_tmpl.format(runner.iter + 1)]
        else:
            filenames = [
                self.filename_tmpl.format(f'{runner.iter + 1}_{k}')
                for k in self.vis_keys
            ]

        # TODO: support format by keys
        for key, filename in zip(self.vis_keys, filenames):
            save_image(
                output_dict[key],
                osp.join(runner.work_dir, self.output_dir, filename),
                nrow=self.nrow,
                padding=self.padding)

    def _rerange_images(self, output_dict):
        for k in output_dict.keys():
            img = output_dict[k]
            if self.rerange:
                output_dict[k] = (img + 1) / 2
            if self.bgr2rgb and img.size(1) == 3:
                img = img[:, [2, 1, 0], ...]
            if img.size(1) == 1:
                img = torch.cat([img, img, img], dim=1)
            img = img.clamp_(0, 1)
            output_dict[k] = img
        return output_dict

    def _unflatten_images(self, runner, output_dict):
        model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model
        camera = model.camera
        H, W = camera.H, camera.W
        for k in output_dict.keys():
            img = output_dict[k]
            assert img.shape[1] == H * W
            img = img.reshape((-1, H, W, img.shape[-1])).permute(0, 3, 1, 2)
            output_dict[k] = img
        return output_dict

    def _check_eval_buffer(self, model, runner_iter):
        if hasattr(model, 'eval_img_buffer'):
            iteration = model.eval_img_buffer_iter
            return iteration == runner_iter
        return False

    def vis_depth_color(self, depth):
        """Visualize depth map with color map.
        Args:
            depth (torch.Tensor): Depth map. Shape like ``[bz, n_points, 1]``.

        Returns:
            torch.Tensor: Colored depth map. Shape like ``[bz, n_points, 3]``.
                Range in ``[0, 1]``.
        """
        # TODO: support scale
        cmap_center = torch.FloatTensor(self._color_map_bincenters)
        cmap = torch.FloatTensor(self._color_map)
        return self._vis_color_map(depth, cmap_center, cmap)

    def vis_depth(self, depth):
        """Visualize depth map with in gray mode.
        Args:
            depth (torch.Tensor): Depth map. Shape like ``[bz, n_points, 1]``.

        Returns:
            torch.Tensor: Gray scale map. Shape like ``[bz, n_points, 3]``.
                Range in ``[0, 1]``.
        """
        return self._vis_gray_img(depth)

    def vis_img(self, img):
        """Visualization function for images.

        Args:
            img (torch.Tensor): Images to visualization. Shape like
                ``[batch_size, n_points, 3]``.
        Returns:
            torch.Tensor: Tensor shape like ``[batch_size, n_points, 3]`` and
                range in ``[0, 1]``.
        """
        assert img.shape[2] == 3, (
            'The input image must shape as \'[batch_size, n_points, 3]\', but '
            f'receive \'{img.shape}\'.')

        if self.rerange:
            img = (img + 1) / 2
        if self.bgr2rgb:
            img = img[..., [2, 1, 0]]
        img = img.clamp_(0, 1)
        return img

    def vis_disp(self, disp):
        """Visualization function for disparity map.
        Args:
            disp (torch.Tensor): Disparity map to visualization. Shape like
                ``[batch_size, n_points, 1]`` or ``[batch_size, n_points]``.

        Returns:
            torch.Tensor: Tensor range in ``[0, 1]``.
        """
        return self._vis_gray_img(disp)

    @staticmethod
    def _vis_gray_img(img):
        """Visualize single channel gray scale image.

        Args:
            img (torch.Tensor):
        """
        img = img.squeeze()
        assert img.ndim == 2, (
            'Input gray scale image should shape as '
            '\'[batch_size, n_points]\' or \'[batch_size, n_points, 1]\', but '
            f'receive \'{img.shape}\'.')
        img_ = img.squeeze()[..., None]
        img_ = img_.clamp_(0, 1)
        img_ = torch.cat([img_] * 3, dim=-1)
        return img_

    @staticmethod
    def _vis_color_map(img, color_map_bin_center, color_map, scale=None):
        """Visualize image with given color map and center.
        Args:
            image (torch.Tensor): Single channel image. Shape like
                ``[batch_size, n_points, 1]`` or ``[batch_size, n_points]``.
            scale (float, optional): Scale factor to norm the depth map to
                ``[0, 1]``. If not passed, the max value in the given depth
                map will be used. Defaults to None.

        Returns:
            torch.Tensor: Colored depth map. Shape like ``[bz, n_points, 3]``.
                Range in ``[0, 1]``.
        """
        batch_size = img.shape[0]

        # convert to torch.Tensor
        if scale is None:
            scale = img.view(batch_size, -1).max(dim=1)[0] + 1e-8
        elif isinstance(scale, (float, int)):
            scale = torch.FloatTensor([float])
        elif isinstance(scale, np.ndarray):
            scale = torch.from_numpy(scale)
        else:
            assert isinstance(scale, torch.Tensor), (
                'Only support \'int\', \'float\', \'np.ndarray\', '
                '\'torch.Tensor\' and \'None\' for \'scale\', but receive '
                f'{type(scale)}.')

        # shape checking for torch.Tensor
        scale_old = scale.clone()  # save for error raising
        scale = scale.squeeze()
        if scale.squeeze().shape == (batch_size, ):
            scale = scale[:, None]
        elif scale.squeeze().shape == ():
            scale = scale[None, None].repeat(batch_size, 1)
        else:
            raise ValueError('Cannot convert input \'scale\' '
                             f'([{scale_old.shape}]) to [{batch_size}, 1].')

        # norm and reshapt to [batch_size, n_points, 1]
        img_norm = img.view(batch_size, -1)
        img_norm = (img_norm / scale).clamp(0, 1)[..., None]

        # [bz, n_points, 1]
        higher_bin = torch.searchsorted(
            color_map_bin_center, img_norm, right=True)
        higher_bin_value = color_map_bin_center[higher_bin]
        lower_bin_value = color_map_bin_center[higher_bin - 1]

        higher_color_value = color_map[higher_bin].squeeze()
        lower_color_value = color_map[higher_bin - 1].squeeze()

        alphas = (img_norm - lower_bin_value) / (
            higher_bin_value - lower_bin_value)

        colors = lower_color_value * (1 - alphas) + higher_color_value * alphas

        # convert colors to tensor and rescale to [0, 1]
        colors = (colors / 255.).clamp_(0, 1)
        return colors

    @torch.no_grad()
    def get_results(self, runner):

        _model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model
        if self._check_eval_buffer(_model, runner.iter):
            _model.collect_img_buffer()
            return _model.eval_img_buffer

        mmcv.print_log(f'Sample {self.num_samples} images for visualization.',
                       'mmgen')

        # TODO: support batch size split and update
        pbar = mmcv.utils.ProgressBar(self.num_samples)
        results = dict()
        for data in self.dataloader:
            # batch_size = data['real_img'].shape[0]
            kwargs = deepcopy(self.kwargs)
            kwargs['mode'] = 'reconstruction'
            kwargs['return_noise'] = True
            prob_dict = runner.model(data, return_loss=False, **kwargs)
            for k, v in prob_dict.items():
                if k in results:
                    results[k].append(v.cpu())
                else:
                    results[k] = [v.cpu()]
            pbar.update()

        for k, v in results.items():
            results[k] = torch.cat(v, dim=0)
        return results

    @master_only
    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        # eval mode
        runner.model.eval()
        # no grad in sampling
        out_dict = self.get_results(runner)
        vis_dict = {k: out_dict[k] for k in self.vis_keys}
        vis_dict = self._unflatten_images(runner, vis_dict)
        vis_dict = self._rerange_images(vis_dict)
        self._save_image(vis_dict, runner)

        # train mode
        runner.model.train()


@HOOKS.register_module()
class GRAFVisHook(VisualizeNeRFSamples):

    _default_vis_key_mapping = dict(
        rgb_final='img',
        depth_final='depth_color',
        disp_final='disp',
        real_imgs='img',
        selected_pixels='img',
        selected_mask='img')
    _allowed_vis_type = ['depth', 'depth_color', 'img', 'disp']
    _buffer_keys = [
        'selected_mask', 'selected_pixels', 'selected_pixels_alpha',
        'real_imgs'
    ]

    def __init__(self,
                 fixed_noise=True,
                 vis_keys_mapping=None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # the sampling noise will be initialized by the first sampling.
        self.sampling_noise = None

        if vis_keys_mapping is not None:
            assert isinstance(
                vis_keys_mapping,
                dict), ('\'vis_key_mapping\' must be a dict, but receive '
                        f'{type(self.vis_key_mapping)}.')
            self.vis_keys_mapping = vis_keys_mapping
        else:
            self.vis_keys_mapping = self._default_vis_key_mapping

        # vis type checking
        for vis_type in self.vis_keys_mapping.values():
            assert vis_type in self._allowed_vis_type, (
                'Only support vislization type in '
                f'\'{self._allowed_vis_type}\', but receive \'{vis_type}\'.')

        # only vis buffer when need
        self.get_buffer = any(
            [k in self._buffer_keys for k in self.vis_keys_mapping])
        self.fixed_noise = fixed_noise

    def _save_image(self, outputs_dict, runner):
        """"""
        vis_keys = list(outputs_dict.keys())
        num_samples = outputs_dict[vis_keys[0]].shape[0]
        num_samples = min(num_samples, self.num_samples)
        samples = []
        for idx in range(num_samples):
            for k in vis_keys:
                samples.append(outputs_dict[k][idx])

        mmcv.mkdir_or_exist(osp.join(runner.work_dir, self.output_dir))

        filename = self.filename_tmpl.format(runner.iter + 1)

        save_image(
            samples,
            osp.join(runner.work_dir, self.output_dir, filename),
            nrow=self.nrow * len(vis_keys),
            padding=self.padding)

    def vis_buffer(self, runner):
        """Visual buffer in model.results. This mainly have real images and
        sampled points.

        Args:
            runner (object): The runner.

        Returns:
            dict: Vis dict
        """
        _model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model

        buffer = runner.outputs['results']
        real_imgs = buffer['real_imgs'].cpu()
        points = buffer['points_selected'].cpu()
        vis_dict = _model.camera.ray_sampler.vis_pixels_on_image(
            real_imgs, points)

        vis_dict = {
            k: v.cpu()
            for k, v in vis_dict.items() if k in self.vis_keys_mapping
        }

        # TODO: maybe we vis this is a better way
        if 'real_imgs' in self.vis_keys_mapping:
            vis_dict['real_imgs'] = real_imgs.cpu()

        return vis_dict

    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        # eval mode
        runner.model.eval()
        # no grad in sampling
        with torch.no_grad():
            outputs_dict = runner.model(
                self.sampling_noise,
                return_loss=False,
                num_batches=self.num_samples,
                return_noise=True,
                **self.kwargs)
            noise_ = outputs_dict['noise_batch']

        # initialize samling noise with the first returned noise
        if self.sampling_noise is None and self.fixed_noise:
            self.sampling_noise = noise_
            # TODO: support sampling pose
            self.sampling_pose = None

        vis_dict = dict()
        for k, v in self.vis_keys_mapping.items():
            # skip buffer imgs
            if k in self._buffer_keys:
                continue
            img = outputs_dict[k]
            vis_func = getattr(self, f'vis_{v}')
            vis_dict[k] = vis_func(img.cpu())
        vis_dict = self._unflatten_images(runner, vis_dict)

        if self.get_buffer:
            vis_dict.update(self.vis_buffer(runner))

        # train mode
        runner.model.train()

        self._save_image(vis_dict, runner)
