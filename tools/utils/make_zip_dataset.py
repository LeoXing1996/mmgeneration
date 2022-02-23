# this file save translated file and write a new config with zip file
import argparse
import io
import os.path as osp
import sys
import zipfile
from copy import deepcopy

import mmcv
from mmcv import Config, print_log
from PIL import Image

# yapf: disable
sys.path.append(osp.abspath(osp.join(__file__, '../../..')))  # isort:skip  # noqa
from mmgen.datasets import (UnconditionalImageDataset,  # isort:skip  # noqa
                            build_dataset)  # isort:skip  # noqa
# yapf: enable


def fliter_pipeline(pipelines, zip_path=None):
    """"""
    allowed_pipelines = ['Resize', 'CenterCropLongEdge']
    shared_pipelines = ['Collect']

    pipeline_use = []
    pipeline_new = []
    for pipeline in pipelines:
        if pipeline['type'] == 'LoadImageFromFile':
            pipeline_use.append(pipeline)
            # convert to zip backend
            load_with_zip_backend = deepcopy(pipeline)
            load_with_zip_backend['io_backend'] = 'ZipBackend'
            load_with_zip_backend['zip_path'] = zip_path
            pipeline_new.append(load_with_zip_backend)
        elif pipeline['type'] in allowed_pipelines:
            pipeline_use.append(deepcopy(pipeline))
        elif pipeline['type'] in shared_pipelines:
            pipeline_use.append(deepcopy(pipeline))
            pipeline_new.append(deepcopy(pipeline))
        else:
            pipeline_new.append(deepcopy(pipeline))
    return pipeline_use, pipeline_new


def parse_data_cfg_and_save(data_cfg, subset, zip_path=None, zip_config=None):
    """Parse the given data config.

    Args:
        data_cfg (dict): The input data config.
        subsets (str): The subset of data.

    Returns:
        tuple: tuple of dict.
    """
    data_cfg_new = deepcopy(data_cfg)

    assert subset in data_cfg, (
        f'Passed subset (\'{subset}\') must in the given data config.')
    subset_cfg = deepcopy(data_cfg[subset])

    # 1. handle repeat dataset
    if subset_cfg['type'] == 'RepeatDataset':
        print_log(f'Fine \'RepeatDataset\' in config of \'{subset}\'. '
                  'Use the original ones.')
        subset_cfg = subset_cfg['dataset']
        use_repeat_dataset = True
    else:
        use_repeat_dataset = False

    # 2. check pipeline
    pipeline = subset_cfg['pipeline']
    pipeline_use, pipeline_new = fliter_pipeline(pipeline, zip_path=zip_path)
    subset_cfg['pipeline'] = pipeline_use

    # 3. update new pipeline use zip loader
    if use_repeat_dataset:
        data_cfg_new[subset]['dataset']['pipeline'] = pipeline_new
        data_cfg_new[subset]['dataset']['imgs_root'] = zip_path
    else:
        data_cfg_new[subset]['pipeline'] = pipeline_new
        data_cfg_new[subset]['imgs_root'] = zip_path

    _cfg = Config(cfg_dict=dict(data=data_cfg_new))
    if zip_config is not None and zip_config.upper() != 'NONE':
        with open(zip_config, 'w', encoding='utf-8') as file:
            file.write(_cfg.pretty_text)
        print_log(
            f'Save new parsed data config to \'{osp.abspath(zip_config)}\'. '
            'You may convert old data config in \'_base_\' to the new one.',
            'mmgen')

    return subset_cfg


def dump_to_zip(zipfile, dataset):

    pbar = mmcv.ProgressBar(len(dataset))
    for idx, data_dict in enumerate(dataset):
        # load img
        if 'real_img' in data_dict:
            img = data_dict['real_img']
        elif 'img' in data_dict:
            img = data_dict['img']
        else:
            raise KeyError('')

        try:
            # convert img to pil image and save
            img_pil = Image.fromarray(img)
        except Exception:
            print_log(
                f'Output type (\'{type(img)}\') is not array, cannot convert '
                'to Image. Please check your pipeline config or dataset '
                'config', 'mmgen')
            exit()
        # init buffer
        image_bits = io.BytesIO()
        img_pil.save(
            image_bits, format='png', compress_level=0, optimize=False)
        zipfile.writestr(f'{idx}.png', image_bits.getbuffer())
        pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make zip loader.')
    parser.add_argument(
        '--imgsdir', type=str, default=None, help='the dir contains images.')
    parser.add_argument(
        '--data-cfg',
        type=str,
        default=None,
        help='the config file for the data pipeline')
    parser.add_argument(
        '--pipeline-cfg',
        type=str,
        default=None,
        help='the config file for the data pipline')
    parser.add_argument(
        '--subset', type=str, default='train', help='the subset to zip')

    parser.add_argument(
        '--size',
        type=int,
        nargs='+',
        default=None,
        help='image size in the data pipeline')

    group_transforms = parser.add_mutually_exclusive_group()
    group_transforms.add_argument(
        '--center-crop',
        action='store_true',
        help='apply centercrop to the original image')
    # TODO:
    # group_transforms.add_argument(
    #     '--center-crop-wide',
    #     action='store_true',
    #     help='apply centercrop wide to the origin image')

    parser.add_argument('--zipdir', type=str, default='./', help='')
    parser.add_argument(
        '--zipname',
        type=str,
        default='dataset.zip',
        help='the name of the output zip file')
    parser.add_argument(
        '--zipconfig',
        type=str,
        default='zip_dataset_config.py',
        help='path to the config of zip data pipeline')

    args = parser.parse_args()

    # dataset pipeline (only be used when args.imgsdir is not None)
    if args.pipeline_cfg is not None:
        pipeline = Config.fromfile(args.pipeline_cfg)
    elif args.imgsdir is not None:
        pipeline = [
            dict(type='LoadImageFromFile', key='real_img', to_rgb=True),
            dict(type='Collect', keys=['real_img'], meta_keys=[]),
        ]
        if args.center_crop:
            pipeline.append(dict(type='CenterCropLongEdge', keys=['img']))
        # TODO:
        # elif args.center_crop_wide:
        #     pipeline.append(type='CenterCropLongEdgeWide', keys=['img'])

        if args.size is not None:
            size = args.size
            pipeline.append(
                dict(
                    type='Resize',
                    keys=['real_img'],
                    scale=size,
                    keep_ratio=False))
        # add collect at the end
        pipeline.append(dict(type='Collect', keys=['real_img'], meta_keys=[]))

    # build dataloader
    if args.imgsdir is not None:
        dataset = UnconditionalImageDataset(args.imgsdir, pipeline)
        # datasets = [UnconditionalImageDataset(args.imgsdir, pipeline)]
    elif args.data_cfg is not None:
        data_config = Config.fromfile(args.data_cfg).data
        subset_config = parse_data_cfg_and_save(
            data_config,
            subset=args.subset,
            zip_path=osp.join(args.zipdir, args.zipname),
            zip_config=args.zipconfig)
        dataset = build_dataset(subset_config)
    else:
        raise RuntimeError('Please provide imgsdir or data_cfg')

    zf = zipfile.ZipFile(
        file=osp.join(args.zipdir, args.zipname),
        mode='w',
        compression=zipfile.ZIP_STORED)

    pbar = mmcv.ProgressBar(len(dataset))
    for idx, data_dict in enumerate(dataset):
        # load img
        if 'real_img' in data_dict:
            img = data_dict['real_img']
        elif 'img' in data_dict:
            img = data_dict['img']
        else:
            raise KeyError('Cannot found key for images in data_dict. '
                           'Only support `real_img` for unconditional '
                           'datasets and `img` for conditional '
                           'datasets.')

        try:
            # convert img to pil image and save
            img_pil = Image.fromarray(img)
        except Exception:
            print_log(
                f'Output type (\'{type(img)}\') is not array, cannot convert '
                'to Image. Please check your pipeline config or dataset '
                'config', 'mmgen')
            exit()
        # init buffer
        image_bits = io.BytesIO()
        img_pil.save(
            image_bits, format='png', compress_level=0, optimize=False)
        zf.writestr(f'{idx}.png', image_bits.getbuffer())
        pbar.update()

    zf.close()
