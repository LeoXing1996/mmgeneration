# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import imageio
import numpy as np
from PIL import Image


class ImageioVideoWriter(object):

    def __init__(self,
                 outfile,
                 fps,
                 save_gif=False,
                 hd_video=False,
                 gif_interval=1,
                 **kwargs):
        """pip install imageio-ffmpeg opencv-python.

        :param outfile:
        :param fps:
        :param save_gif:
        :param hd_video:
        :param quality:
          Video output quality. Default is 5. Uses variable bit rate. Highest quality is 10, lowest is 0.
        [https://imageio.readthedocs.io/en/stable/format_ffmpeg.html?highlight=codec#parameters-for-saving](https://imageio.readthedocs.io/en/stable/format_ffmpeg.html?highlight=codec#parameters-for-saving)  # noqa

        :param kwargs:
        """

        self.video_file = outfile
        outfile = Path(outfile)
        self.gif_file = f'{outfile.parent}/{outfile.stem}.gif'
        self.save_gif = save_gif
        self.gif_interval = gif_interval

        self.counter = 0

        self.video = imageio.get_writer(f'{outfile}', fps=fps)
        if hd_video:
            outfile_hd = f'{outfile.parent}/{outfile.stem}_hd.mp4'
            self.video_file_hd = outfile_hd
            self.video_hd = imageio.get_writer(
                outfile_hd, mode='I', fps=fps, codec='libx264', bitrate='16M')
        else:
            self.video_hd = None

        if self.save_gif:
            self.gif_out = imageio.get_writer(
                self.gif_file, fps=fps // gif_interval)

    def write(self, image, dst_size=None, **kwargs):
        if dst_size is not None:
            w, h = self._get_size(
                w=image.size[0],
                h=image.size[1],
                dst_size=dst_size,
                for_min_edge=False)
            image = image.resize((w, h), Image.LANCZOS)
        img_np = np.array(image)
        self.video.append_data(img_np)
        if self.video_hd is not None:
            self.video_hd.append_data(img_np)

        if self.save_gif:
            if self.counter % self.gif_interval == 0:
                self.gif_out.append_data(img_np)

        self.counter += 1

    def _get_size(self, w, h, dst_size, for_min_edge=True):
        if for_min_edge:
            edge = min(w, h)
        else:
            edge = max(w, h)

        w = int(dst_size / edge * w)
        h = int(dst_size / edge * h)
        return w, h
