# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from copy import deepcopy

from torch.utils.data import Dataset

from .pipelines import Compose


class CustomNeRFDataset(Dataset, metaclass=ABCMeta):
    """Custom dataset for nerf-based methods."""

    IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                      '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')

    def __init__(self, dataroot, pipeline, split, skip_per_image, test_mode,
                 load_in_memory):
        super().__init__()
        self.data_root = dataroot
        self.pipeline = Compose(pipeline)
        self.split = split
        # TODO: this feature will be supported later.
        self.load_in_memory = load_in_memory

        # this festure seems no use
        self.test_mode = test_mode
        self.skip_per_image = skip_per_image
        self.data_info = self.load_annotations()
        self.is_train = split == 'train'

    @abstractmethod
    def load_annotations(self):
        """Load annotation data for NeRF data."""
        pass

    def prepare_rays(self, data_info):
        """Prepare camera and rays for rendering.
        Args:
            data_info (dict): Dict contains requiring information.

        Returns:
            dict: Updated dict contains full camera parameters required
                for rendering.
        """
        if self.load_in_memory:
            return data_info

        # apply coordinates transformation
        ray_info = {k: data_info[v] for k, v in self.DATA_INFO_MAPPING.items()}
        if not self.is_train:
            ray_info['random_sample'] = False

        rays_dict = self.camera.prepare_render_rays(**ray_info)
        data_info.update(rays_dict)
        return data_info

    def prepare_data(self, idx):
        """Prepare data for the network.
        Args:
            idx (int): Index of the batch data.

        Returns:
            dict: Returned batch.
        """
        data_info_ = deepcopy(self.data_info[idx])
        data_info_ = self.pipeline(data_info_)
        # data_info_ = self.prepare_rays(data_info_)
        return data_info_

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        return self.prepare_data(idx)
