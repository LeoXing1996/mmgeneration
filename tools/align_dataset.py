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
        mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
    mm_crop = CenterCropLongEdge(keys=['img'])
    mm_norm = Normalize(**img_norm_cfg)

    # mm_toTensor = ImageToTensor(keys=['img'])

    def mm_pipeline(image_path, use_fileHandler=False, no_norm=False):
        if use_fileHandler:
            image_dict = mm_load(
                dict(img_info=dict(filename=image_path), img_prefix=None))
        else:
            image = np.array(Image.open(image_path))
            image_dict = dict(img=image)

        image_dict = ddpm_resize(image_dict)

        image_dict = mm_crop(image_dict)

        if not no_norm:
            # subtract then divide
            image_dict = mm_norm(image_dict)

        return image_dict

    def off_pipeline(image_path, no_norm=False):
        pil_image = Image.open(image_path)

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

        arr = np.array(pil_image.convert('RGB'))
        crop_y = (arr.shape[0] - resolution) // 2
        crop_x = (arr.shape[1] - resolution) // 2
        arr = arr[crop_y:crop_y + resolution, crop_x:crop_x + resolution]

        if not no_norm:
            # divide then abstract
            arr = arr.astype(np.float32) / 127.5 - 1
        return arr

    mm_res = mm_pipeline(img_path, no_norm=True)['img']
    off_res = off_pipeline(img_path, no_norm=True)
    print('No Norm: ', (mm_res == off_res).all())

    mm_res = mm_pipeline(img_path)['img']
    off_res = off_pipeline(img_path)
    print('With Norm', (mm_res == off_res).all())

    mm_res = mm_pipeline(img_path, use_fileHandler=True, no_norm=True)['img']
    off_res = off_pipeline(img_path, no_norm=True)
    print('No Norm, MMFileHandler', (mm_res == off_res).all())


if __name__ == '__main__':
    main()
