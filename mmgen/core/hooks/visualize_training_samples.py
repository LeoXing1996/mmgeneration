# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from copy import deepcopy

import mmcv
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
class VisualizeReconstructionSamples(Hook):
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

    def __init__(self,
                 output_dir,
                 dataloader,
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
