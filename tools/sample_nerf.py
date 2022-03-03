# Copyright (c) OpenMMLab. All rights reserved.
import json
import math
from argparse import ArgumentParser

import numpy as np
import torch
import tqdm
from easydict import EasyDict
from mmcv.runner import set_random_seed as set_random_seed_mmcv
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from video_util import ImageioVideoWriter

from mmgen.apis import init_model
from mmgen.models import (get_circle_camera_pos_and_lookup,
                          get_translate_circle_camera_pos_and_lookup)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument(
        '--traj',
        type=str,
        choices=['circle_near_far', 'translate_circel_near_far', 'random'])
    parser.add_argument('--fix-noise', action='store_true')
    parser.add_argument('--fix-pose', action='store_true')
    parser.add_argument('--num-samples', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--fov', type=float, default=12)
    parser.add_argument('--max-fov', type=float, default=20)
    parser.add_argument('--alpha-pi-div', type=float, default=15)
    parser.add_argument('--translate-dist', type=float, default=0.04)

    parser.add_argument('--fps', type=int, default=40)
    parser.add_argument('--psi', type=float, default=0.95)
    parser.add_argument('--num_frames', type=int, default=70)
    parser.add_argument('--num_steps', type=int, default=12)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--num_samples_translate', type=int, default=30)
    parser.add_argument('--curr_file', default='./tools/meta.json')

    parser.add_argument(
        '--return_nerf', action='store_true', help='Get the NeRF results')
    parser.add_argument('--outdir', default='./')
    parser.add_argument('--output_tmp', default='{}.mp4')

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)

    return parser.parse_args()


def get_metadata_from_json(metafile,
                           return_raw=False,
                           image_size=256,
                           num_steps=12,
                           psi=0.9,
                           fov=12,
                           v_stddev=0,
                           h_stddev=0,
                           last_back=False,
                           nerf_noise=0):
    with open(metafile, 'r') as f:
        curriculum = json.load(f)
        curriculum = EasyDict(curriculum)

    if return_raw:
        return curriculum

    curriculum['img_size'] = image_size
    curriculum['num_steps'] = num_steps
    curriculum['psi'] = psi
    curriculum['fov'] = fov
    curriculum['v_stddev'] = v_stddev
    curriculum['h_stddev'] = h_stddev
    curriculum['last_back'] = last_back
    curriculum['nerf_noise'] = nerf_noise
    return curriculum


def to_pil(frame):
    frame = (frame.squeeze() + 1) * 0.5
    frame_pil = to_pil_image(frame)
    return frame_pil


def get_size(w, h, dst_size, for_min_edge=True):
    if for_min_edge:
        edge = min(w, h)
    else:
        edge = max(w, h)

    w = int(dst_size / edge * w)
    h = int(dst_size / edge * h)
    return w, h


def merge_image_pil(image_list,
                    nrow: int = 8,
                    saved_file=None,
                    pad=0,
                    pad_color='black',
                    dst_size=None):

    max_h, max_w = 0, 0
    ncol = (len(image_list) + nrow - 1) // nrow
    for img in image_list:
        max_h = max(max_h, img.size[1])
        max_w = max(max_w, img.size[0])

    H = ncol * max_h + pad * (ncol - 1)
    W = nrow * max_w + pad * (nrow - 1)
    merged_image = Image.new(mode='RGB', size=(W, H), color=pad_color)

    for idx, img in enumerate(image_list):
        row = idx // nrow
        col = idx % nrow
        merged_image.paste(img, (col * (max_w + pad), row * (max_h + pad)))

    if dst_size is not None:
        out_w, out_h = get_size(
            w=merged_image.size[0],
            h=merged_image.size[1],
            dst_size=dst_size,
            for_min_edge=False)
        merged_image = merged_image.resize((out_w, out_h), Image.LANCZOS)

    if saved_file is not None:
        merged_image.save(saved_file)
    return merged_image


def main():
    args = parse_args()

    set_random_seed_mmcv(args.seed, deterministic=True)

    device = args.device
    # device = 'cpu'
    model = init_model(args.config, checkpoint=args.checkpoint, device=device)

    fov = args.fov
    max_fov = args.max_fov
    alpha_pi_div = args.alpha_pi_div
    translate_dist = args.translate_dist

    num_frames = args.num_frames
    fps = args.fps
    num_samples_translate = args.num_samples_translate
    forward_points = 256

    if args.traj == 'circle_near_far':
        xyz, lookup, yaws, pitchs = get_circle_camera_pos_and_lookup(
            alpha=math.pi / alpha_pi_div, num_samples=num_frames, periods=2)
        xyz = torch.from_numpy(xyz).to(device)
        lookup = torch.from_numpy(lookup).to(device)
        fov_list = []
        for idx, t in enumerate(np.linspace(0, 1, num_frames)):
            fov_list.append(fov + t * (max_fov - fov))
        fov_list.extend(fov_list[::-1])
    elif args.traj == 'translate_circle_near_far':
        xyz, lookup, yaws, pitchs, num_samples_translate = \
            get_translate_circle_camera_pos_and_lookup(
                num_samples_translate=num_samples_translate,
                translate_dist=translate_dist,
                alpha=math.pi / alpha_pi_div,
                num_samples=num_frames,
                periods=2)
        xyz = torch.from_numpy(xyz).to(device)
        lookup = torch.from_numpy(lookup).to(device)
        fov_list = [fov] * num_samples_translate * 2
        for idx, t in enumerate(np.linspace(0, 1, num_frames)):
            fov_list.append(fov + t * (max_fov - fov))
        fov_list.extend(fov_list[-num_frames:][::-1])
        assert len(fov_list) == len(xyz)
    elif args.traj == 'circle':
        xyz, lookup, yaws, pitchs = get_circle_camera_pos_and_lookup(
            alpha=math.pi / alpha_pi_div, num_samples=num_frames, periods=2)
        xyz = torch.from_numpy(xyz).to(device)
        lookup = torch.from_numpy(lookup).to(device)
        fov_list = [fov] * len(xyz)
    elif args.traj == 'yaw':
        # TODO:
        # xyz, lookup, yaws, pitchs = get_yaw_pitch_by_xyz(
        #     num_samples=num_frames, )
        # xyz = torch.from_numpy(xyz).to(device)
        # lookup = torch.from_numpy(lookup).to(device)
        # fov_list = [fov] * len(xyz)
        pass

    outdir = args.outdir
    output_name = args.output_tmp.format(args.seed)
    video_f = ImageioVideoWriter(f'{outdir}/{output_name}', fps=fps)
    curriculum = get_metadata_from_json(
        args.curr_file,
        num_steps=args.num_steps,
        image_size=args.image_size,
        psi=args.psi)

    # manually set device for cipd-3d generator
    model.generator.device = device
    zs = model.generator.get_zs(1)
    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(xyz))):
            curriculum['h_mean'] = 0
            curriculum['v_mean'] = 0
            curriculum['h_stddev'] = 0
            curriculum['v_stddev'] = 0

            cur_camera_pos = xyz[[idx]]
            cur_camera_lookup = lookup[[idx]]
            # yaw = yaws[idx]
            # pitch = pitchs[idx]
            # fov = fov_list[idx]
            curriculum['fov'] = fov

            frame, depth_map = \
                model.generator.forward_camera_pos_and_lookup(
                    zs=zs,
                    return_aux_img=args.return_nerf,
                    return_nerf_fea=args.return_nerf,
                    forward_points=forward_points ** 2,
                    camera_pos=cur_camera_pos,
                    camera_lookup=cur_camera_lookup,
                    **curriculum)
            if args.return_nerf:
                merged_frame = merge_image_pil([
                    to_pil(frame[0].cpu()),
                    to_pil(frame[1].cpu()),
                    to_pil(frame[2].cpu())
                ],
                                               nrow=3)
            else:
                frame_pil = to_pil(frame.cpu())
                merged_frame = merge_image_pil(
                    [frame_pil],
                    nrow=2,
                )

            video_f.write(merged_frame)


if __name__ == '__main__':
    main()
