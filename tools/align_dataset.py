from copy import deepcopy

import numpy as np
from mmcls.datasets.pipelines.loading import LoadImageFromFile
from mmcls.datasets.pipelines.transforms import Normalize
from PIL import Image

from mmgen.datasets.pipelines.augmentation import (CenterCropLongEdge,
                                                   DDPMResize)

# from mmgen.datasets.pipelines.augmentation import MultiScaleResize


def main():

    img_path = ('/space0/home/xingzn/mmgen_dev/DDPM/data/'
                'imagenet/train/n01443537/n01443537_9977.JPEG')
    # img_path = ('/space0/home/xingzn/code/improved-diffusion/data/'
    #             'cifar_train/car_10251.png')
    mm_load = LoadImageFromFile()
    # mm_resize = MultiScaleResize(
    #     keys=['img'],
    #     scale=128,
    #     backend='pillow',
    #     interpolation='box',
    #     last_interpolation='bicubic')
    ddpm_resize = DDPMResize(keys=['img'], resolution=128)
    img_norm_cfg = dict(
        mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
    mm_crop = CenterCropLongEdge(keys=['img'])
    mm_norm = Normalize(**img_norm_cfg)

    # mm_toTensor = ImageToTensor(keys=['img'])

    def mm_pipeline(image_path, use_fileHandler=False, no_norm=False):
        res_dict = dict()
        if use_fileHandler:
            image_dict = mm_load(
                dict(img_info=dict(filename=image_path), img_prefix=None))
            # bgr to rgb manually here
            image_dict['img'] = image_dict['img'][..., (2, 1, 0)]
        else:
            image = np.array(Image.open(image_path))
            image_dict = dict(img=image)
        res_dict['img_loading'] = deepcopy(image_dict['img'])

        image_dict = ddpm_resize(image_dict)
        res_dict['img_resize'] = deepcopy(image_dict['img'])

        image_dict = mm_crop(image_dict)
        res_dict['img_crop'] = deepcopy(image_dict['img'])

        if not no_norm:
            # subtract then divide
            image_dict = mm_norm(image_dict)
            res_dict['img_norm'] = deepcopy(image_dict['img'])
        res_dict['img_final'] = deepcopy(image_dict['img'])

        return image_dict, res_dict

    def off_pipeline(image_path, no_norm=False):
        res_dict = dict()
        pil_image = Image.open(image_path)
        res_dict['img_loading'] = np.array(pil_image)

        resolution = 128
        while min(*pil_image.size) >= 2 * resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
            print(pil_image.size)

        scale = resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size),
            resample=Image.BICUBIC)
        print(pil_image.size)
        res_dict['img_resize'] = deepcopy(np.array(pil_image))

        arr = np.array(pil_image.convert('RGB'))
        crop_y = (arr.shape[0] - resolution) // 2
        crop_x = (arr.shape[1] - resolution) // 2
        arr = arr[crop_y:crop_y + resolution, crop_x:crop_x + resolution]
        res_dict['img_crop'] = deepcopy(arr)

        if not no_norm:
            # divide then subtract
            arr = arr.astype(np.float32) / 127.5 - 1
            res_dict['img_norm'] = deepcopy(arr)
        res_dict['img_final'] = deepcopy(arr)
        return arr, res_dict

    print('======= NO NORM =======')
    mm_res, mm_dict = mm_pipeline(img_path, no_norm=True)
    off_res, off_dict = off_pipeline(img_path, no_norm=True)
    show_diff(mm_dict, off_dict)

    print('======= WITH NORM =======')
    mm_res, mm_dict = mm_pipeline(img_path)
    off_res, off_dict = off_pipeline(img_path)
    show_diff(mm_dict, off_dict)

    print('======= WITH NORM + WITH FILDHANDLER=======')
    mm_res, mm_dict = mm_pipeline(img_path, use_fileHandler=True, no_norm=True)
    off_res, off_dict = off_pipeline(img_path, no_norm=True)
    show_diff(mm_dict, off_dict)


def show_diff(mm_dict, off_dict):
    for k in mm_dict.keys():
        diff = np.abs(mm_dict[k] - off_dict[k])
        print(f'{k}: MAX Error: {diff.max()}, MEAN: {diff.mean()}, '
              f'STD: {diff.std()}')
        if diff.max() == 255:
            coor = np.nonzero(diff == diff.max())
            n_points = len(coor[0])
            coor_list = [[
                int(coor[0][idx]),
                int(coor[1][idx]),
                int(coor[2][idx])
            ] for idx in range(n_points)]
            print(f'MAX COOR: {coor_list}')


if __name__ == '__main__':
    main()
