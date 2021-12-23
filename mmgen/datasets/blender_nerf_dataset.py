# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import numpy as np

from .builder import DATASETS
from .custom_nerf_dataset import CustomNeRFDataset


@DATASETS.register_module()
class BlenderDataset(CustomNeRFDataset):
    """Dataset for Blender lego.

    ├── test
    │   ├── r_0_depth_0001.png
    │   ├── r_0_normal_0001.png
    │   ├── r_0.png
    │   ...
    ├── train
    │   ├── r_0.png
    │   ├── r_10.png
    │   ├── r_11.png
    │   ...
    ├── transforms_test.json
    ├── transforms_train.json
    ├── transforms_val.json
    └── val
         ├── r_0.png
         ├── r_10.png
         ├── r_11.png
         ├── ...
    """

    DATA_INFO_MAPPING = dict(
        transform_matrix='transform_matrix', img='real_img')

    def __init__(self,
                 dataroot,
                 pipeline,
                 split='train',
                 skip_per_image=None,
                 test_mode=False,
                 load_in_memory=False):
        super().__init__(dataroot, pipeline, split, skip_per_image, test_mode,
                         load_in_memory)

    def _load_image_from_dir(self):
        """Load annotations."""
        meta_path = osp.join(self.data_root, f'transforms_{self.split}.json')

        data_infos = dict()
        with open(meta_path, 'r') as file:
            meta = json.load(file)

        true_idx = 0
        for idx, frame in enumerate(meta['frames']):
            if (self.skip_per_image is not None
                    and (idx % self.skip_per_image) != 0):
                continue
            file_name = osp.join(self.data_root, frame['file_path'] + '.png')
            transform_matrix = np.array(frame['transform_matrix']).astype(
                np.float32)
            data_infos[true_idx] = dict(
                real_img_path=file_name, transform_matrix=transform_matrix)
            true_idx += 1

        return data_infos

    def load_annotations(self):
        return self._load_image_from_dir()
