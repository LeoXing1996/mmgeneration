import argparse
import os
import os.path as osp
import sys

import mmcv
import numpy as np
from mmcv import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import init_model, init_dataset, sample_nerf_model  # isort:skip  # noqa

# yapf: enable

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


def parse_args():
    parser = argparse.ArgumentParser(description='NeRF demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--subset',
        default='test',
        help='Which subset of the dataset used to generate images.')
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/nerf_samples.gif',
        help='path to save nerf samples, image nad gif are both supported')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CUDA device id')
    # TODO: we may provide a api to use a specific camera -->
    # over camera cfg here

    # args for inference/sampling
    parser.add_argument(
        '--num-samples',
        type=int,
        default=-1,
        help=('The total number of samples. If \'-1\' is passed, all samples '
              'in the dataset will be samples'))
    parser.add_argument(
        '--sample-model',
        type=str,
        default='ema',
        help='Which model to use for sampling')
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')

    # args for image grid
    parser.add_argument(
        '--padding', type=int, default=0, help='Padding in the image grid.')
    parser.add_argument(
        '--nrow',
        type=int,
        default=6,
        help='Number of images displayed in each row of the grid')

    # args for output type
    parser.add_argument(
        '--is-rgb',
        action='store_true',
        help=('If true, color channels will not be permuted, This option is '
              'useful when inference model trained with rgb images.'))
    parser.add_argument('--fps', default=24, help=())

    args = parser.parse_args()
    return args


def create_gif(results, gif_name, fps=0.1):
    """Create gif through imageio.

    Args:
        frames (torch.Tensor): Image frames, shape like [bz, 3, H, W].
        gif_name (str): Saved gif name
        duration (int): Display interval (s). Default: 0.1.
    """
    import imageio
    frames_list = []
    for frame in results:
        frames_list.append(
            (frame.permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8))
    if imageio is None:
        raise RuntimeError('imageio is not installed,'
                           'Please use “pip install imageio” to install')
    imageio.mimsave(gif_name, frames_list, 'GIF', fps=fps)


def main():
    args = parse_args()
    model = init_model(
        args.config, checkpoint=args.checkpoint, device=args.device)

    dataset = init_dataset(args.config, args.subset)

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    if args.num_samples == -1:
        args.num_samples = len(dataset)

    results = sample_nerf_model(model, dataset, args.num_samples,
                                args.sample_model, **args.sample_cfg)

    if not args.is_rgb:
        results = results[:, [2, 1, 0]]

    try:
        nerf_act_cfg = model._neural_renderer_cfg['rgb_act_cfg']
        if nerf_act_cfg['type'] != 'Sigmoid':
            results = (results + 1.) / 2.
    except KeyError:
        # judge results by min value
        if results.min() < 0:
            results = (results + 1.) / 2.

    mmcv.mkdir_or_exist(os.path.dirname(args.save_path))

    # save images / gif
    # TODO: support save mps file
    suffix = osp.splitext(args.save_path)[-1]
    if suffix in IMG_EXTENSIONS:
        utils.save_image(
            results, args.save_path, nrow=args.nrow, padding=args.padding)
    elif suffix == '.gif':
        create_gif(results, args.save_path, args.fps)
    else:
        img_types = ', '.join(IMG_EXTENSIONS)
        raise ValueError(f'Unsupport suffix. Only support {img_types}, .gif '
                         f'and for results saving, but receive {suffix}.')


if __name__ == '__main__':
    main()
