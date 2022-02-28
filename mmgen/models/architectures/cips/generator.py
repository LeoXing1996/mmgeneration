# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from generator_style_mapping import \
    MultiHeadMappingNetwork as GenMultiHeadMappingNetwork

from mmgen.models.builder import MODULES
from .camera_utils import fancy_integration
from .comm_utils import (gather_points, get_world_points_and_direction,
                         scatter_points)
from .inr_nerf_base import GeneratorNerfINR
from .module import (FiLMLayer, LinearBlock, SinBlock, ToRGB, UniformBoxWarp,
                     frequency_init, kaiming_leaky_init)


@MODULES.register_module()
class CIPS3DGenerator(GeneratorNerfINR):

    def __init__(self,
                 z_dim,
                 nerf_cfg,
                 mapping_nerf_cfg,
                 inr_cfg,
                 mapping_inr_cfg,
                 device='cuda',
                 **kwargs):
        super().__init__()

        self.epoch = 0
        self.step = 0
        self.z_dim = z_dim
        self.device = device

        self.module_name_list = []
        # nerf_net
        self.siren = NeRFNetwork(**nerf_cfg)
        self.module_name_list.append('siren')

        self.mapping_network_nerf = GenMultiHeadMappingNetwork(
            **{
                **mapping_nerf_cfg, 'head_dim_dict': self.siren.style_dim_dict
            })
        self.module_name_list.append('mapping_network_nerf')

        # inr_net
        self.inr_net = CIPSGenerator(**{
            **inr_cfg, 'input_dim': self.siren.rgb_dim
        })
        self.module_name_list.append('inr_net')

        self.mapping_network_inr = GenMultiHeadMappingNetwork(
            **{
                **mapping_inr_cfg, 'head_dim_dict': self.inr_net.style_dim_dict
            })
        self.module_name_list.append('mapping_network_inr')

        self.aux_to_rbg = nn.Sequential(
            nn.Linear(self.siren.rgb_dim, 3), nn.Tanh())
        self.aux_to_rbg.apply(frequency_init(25))
        self.module_name_list.append('aux_to_rbg')

        self.filters = nn.Identity()

    def forward(self,
                zs,
                img_size,
                fov,
                ray_start,
                ray_end,
                num_steps,
                h_stddev,
                v_stddev,
                hierarchical_sample,
                h_mean=math.pi * 0.5,
                v_mean=math.pi * 0.5,
                psi=1,
                sample_dist=None,
                lock_view_dependence=False,
                clamp_mode='relu',
                nerf_noise=0.0,
                white_back=False,
                last_back=False,
                return_aux_img=False,
                grad_points=None,
                forward_points=None,
                **kwargs):
        """Generates images from a noise vector, rendering parameters, and
        camera distribution. Uses the hierarchical sampling scheme described in
        NeRF.

        :param z: (b, z_dim)
        :param img_size:
        :param fov: face: 12
        :param ray_start: face: 0.88
        :param ray_end: face: 1.12
        :param num_steps: face: 12
        :param h_stddev: face: 0.3
        :param v_stddev: face: 0.155
        :param h_mean: face: pi/2
        :param v_mean: face: pi/2
        :param hierarchical_sample: face: true
        :param psi: [0, 1]
        :param sample_dist: mode for sample_camera_positions, face: 'gaussian'
        :param lock_view_dependence: face: false
        :param clamp_mode: face: 'relu'
        :param nerf_noise:
        :param last_back: face: false
        :param white_back: face: false
        :param kwargs:
        :return:
        - pixels: (b, 3, h, w)
        - pitch_yaw: (b, 2)
        """

        # mapping network
        style_dict = self.mapping_network(**zs)

        if psi < 1:
            avg_styles = self.generate_avg_frequencies(device=self.device)
            style_dict = self.get_truncated_freq_phase(
                raw_style_dict=style_dict,
                avg_style_dict=avg_styles,
                raw_lambda=psi)

        if grad_points is not None and grad_points < img_size**2:
            imgs, pitch_yaw = self.part_grad_forward(
                style_dict=style_dict,
                img_size=img_size,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end,
                num_steps=num_steps,
                h_stddev=h_stddev,
                v_stddev=v_stddev,
                h_mean=h_mean,
                v_mean=v_mean,
                hierarchical_sample=hierarchical_sample,
                sample_dist=sample_dist,
                lock_view_dependence=lock_view_dependence,
                clamp_mode=clamp_mode,
                nerf_noise=nerf_noise,
                white_back=white_back,
                last_back=last_back,
                return_aux_img=return_aux_img,
                grad_points=grad_points,
            )
            return imgs, pitch_yaw
        else:
            imgs, pitch_yaw = self.whole_grad_forward(
                style_dict=style_dict,
                img_size=img_size,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end,
                num_steps=num_steps,
                h_stddev=h_stddev,
                v_stddev=v_stddev,
                h_mean=h_mean,
                v_mean=v_mean,
                hierarchical_sample=hierarchical_sample,
                sample_dist=sample_dist,
                lock_view_dependence=lock_view_dependence,
                clamp_mode=clamp_mode,
                nerf_noise=nerf_noise,
                white_back=white_back,
                last_back=last_back,
                return_aux_img=return_aux_img,
                forward_points=forward_points,
            )
            return imgs, pitch_yaw

    def get_batch_style_dict(self, b, style_dict):
        ret_style_dict = {}
        for name, style in style_dict.items():
            ret_style_dict[name] = style[[b]]
        return ret_style_dict

    def whole_grad_forward(self,
                           style_dict,
                           img_size,
                           fov,
                           ray_start,
                           ray_end,
                           num_steps,
                           h_stddev,
                           v_stddev,
                           h_mean,
                           v_mean,
                           hierarchical_sample,
                           sample_dist=None,
                           lock_view_dependence=False,
                           clamp_mode='relu',
                           nerf_noise=0.0,
                           white_back=False,
                           last_back=False,
                           return_aux_img=True,
                           forward_points=None,
                           camera_pos=None,
                           camera_lookup=None,
                           up_vector=None):
        device = self.device
        # batch_size = z.shape[0]
        batch_size = list(style_dict.values())[0].shape[0]

        if forward_points is not None:
            # stage forward
            with torch.no_grad():
                num_points = img_size**2
                inr_img_output = torch.zeros((batch_size, num_points, 3),
                                             device=device)
                if return_aux_img:
                    aux_img_output = torch.zeros((batch_size, num_points, 3),
                                                 device=device)
                pitch_list = []
                yaw_list = []
                for b in range(batch_size):
                    (
                        transformed_points,
                        transformed_ray_directions_expanded,
                        transformed_ray_origins,
                        transformed_ray_directions,
                        z_vals,
                        pitch,
                        yaw,
                    ) = get_world_points_and_direction(
                        batch_size=1,
                        num_steps=num_steps,
                        img_size=img_size,
                        fov=fov,
                        ray_start=ray_start,
                        ray_end=ray_end,
                        h_stddev=h_stddev,
                        v_stddev=v_stddev,
                        h_mean=h_mean,
                        v_mean=v_mean,
                        sample_dist=sample_dist,
                        lock_view_dependence=lock_view_dependence,
                        device=device,
                        camera_pos=camera_pos,
                        camera_lookup=camera_lookup,
                        up_vector=up_vector,
                    )
                    pitch_list.append(pitch)
                    yaw_list.append(yaw)

                    transformed_points = rearrange(
                        transformed_points,
                        'b (h w s) c -> b (h w) s c',
                        h=img_size,
                        s=num_steps)
                    transformed_ray_directions_expanded = rearrange(
                        transformed_ray_directions_expanded,
                        'b (h w s) c -> b (h w) s c',
                        h=img_size,
                        s=num_steps)
                    head = 0
                    while head < num_points:
                        tail = head + forward_points
                        cur_style_dict = self.get_batch_style_dict(
                            b=b, style_dict=style_dict)
                        cur_inr_img, cur_aux_img = self.points_forward(
                            style_dict=cur_style_dict,
                            transformed_points=transformed_points[:, head:
                                                                  tail],  # noqa
                            transformed_ray_directions_expanded=  # noqa
                            transformed_ray_directions_expanded[:, head:
                                                                tail],  # noqa
                            num_steps=num_steps,
                            hierarchical_sample=hierarchical_sample,
                            z_vals=z_vals[:, head:tail],
                            clamp_mode=clamp_mode,
                            nerf_noise=nerf_noise,
                            transformed_ray_origins=  # noqa
                            transformed_ray_origins[:, head:tail],
                            transformed_ray_directions=  # noqa
                            transformed_ray_directions[:, head:tail],
                            white_back=white_back,
                            last_back=last_back,
                            return_aux_img=return_aux_img,
                        )
                        inr_img_output[b:b + 1, head:tail] = cur_inr_img
                        if return_aux_img:
                            aux_img_output[b:b + 1, head:tail] = cur_aux_img
                        head += forward_points
                inr_img = inr_img_output
                if return_aux_img:
                    aux_img = aux_img_output
                pitch = torch.cat(pitch_list, dim=0)
                yaw = torch.cat(yaw_list, dim=0)
        else:
            (
                transformed_points,
                transformed_ray_directions_expanded,
                transformed_ray_origins,
                transformed_ray_directions,
                z_vals,
                pitch,
                yaw,
            ) = get_world_points_and_direction(
                batch_size=batch_size,
                num_steps=num_steps,
                img_size=img_size,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end,
                h_stddev=h_stddev,
                v_stddev=v_stddev,
                h_mean=h_mean,
                v_mean=v_mean,
                sample_dist=sample_dist,
                lock_view_dependence=lock_view_dependence,
                device=device,
                camera_pos=camera_pos,
                camera_lookup=camera_lookup,
            )

            transformed_points = rearrange(
                transformed_points,
                'b (h w s) c -> b (h w) s c',
                h=img_size,
                s=num_steps)
            transformed_ray_directions_expanded = rearrange(
                transformed_ray_directions_expanded,
                'b (h w s) c -> b (h w) s c',
                h=img_size,
                s=num_steps)
            inr_img, aux_img = self.points_forward(
                style_dict=style_dict,
                transformed_points=transformed_points,
                transformed_ray_directions_expanded=  # noqa
                transformed_ray_directions_expanded,
                num_steps=num_steps,
                hierarchical_sample=hierarchical_sample,
                z_vals=z_vals,
                clamp_mode=clamp_mode,
                nerf_noise=nerf_noise,
                transformed_ray_origins=transformed_ray_origins,
                transformed_ray_directions=transformed_ray_directions,
                white_back=white_back,
                last_back=last_back,
                return_aux_img=return_aux_img,
            )

        inr_img = rearrange(inr_img, 'b (h w) c -> b c h w', h=img_size)
        inr_img = self.filters(inr_img)
        pitch_yaw = torch.cat([pitch, yaw], -1)

        if return_aux_img:
            aux_img = rearrange(aux_img, 'b (h w) c -> b c h w', h=img_size)

            imgs = torch.cat([inr_img, aux_img])
            pitch_yaw = torch.cat([pitch_yaw, pitch_yaw])
        else:
            imgs = inr_img

        return imgs, pitch_yaw

    def part_grad_forward(self,
                          style_dict,
                          img_size,
                          fov,
                          ray_start,
                          ray_end,
                          num_steps,
                          h_stddev,
                          v_stddev,
                          h_mean,
                          v_mean,
                          hierarchical_sample,
                          sample_dist=None,
                          lock_view_dependence=False,
                          clamp_mode='relu',
                          nerf_noise=0.0,
                          white_back=False,
                          last_back=False,
                          return_aux_img=True,
                          grad_points=None,
                          camera_pos=None,
                          camera_lookup=None):
        device = self.device
        batch_size = list(style_dict.values())[0].shape[0]
        (
            transformed_points,
            transformed_ray_directions_expanded,
            transformed_ray_origins,
            transformed_ray_directions,
            z_vals,
            pitch,
            yaw,
        ) = get_world_points_and_direction(
            batch_size=batch_size,
            num_steps=num_steps,
            img_size=img_size,
            fov=fov,
            ray_start=ray_start,
            ray_end=ray_end,
            h_stddev=h_stddev,
            v_stddev=v_stddev,
            h_mean=h_mean,
            v_mean=v_mean,
            sample_dist=sample_dist,
            lock_view_dependence=lock_view_dependence,
            device=device,
            camera_pos=camera_pos,
            camera_lookup=camera_lookup,
        )

        transformed_points = rearrange(
            transformed_points,
            'b (h w s) c -> b (h w) s c',
            h=img_size,
            s=num_steps)
        transformed_ray_directions_expanded = rearrange(
            transformed_ray_directions_expanded,
            'b (h w s) c -> b (h w) s c',
            h=img_size,
            s=num_steps)

        num_points = transformed_points.shape[1]
        assert num_points > grad_points
        rand_idx = torch.randperm(num_points, device=device)
        idx_grad = rand_idx[:grad_points]
        idx_no_grad = rand_idx[grad_points:]

        inr_img_grad, aux_img_grad = self.points_forward(
            style_dict=style_dict,
            transformed_points=transformed_points,
            transformed_ray_directions_expanded=  # noqa
            transformed_ray_directions_expanded,
            num_steps=num_steps,
            hierarchical_sample=hierarchical_sample,
            z_vals=z_vals,
            clamp_mode=clamp_mode,
            nerf_noise=nerf_noise,
            transformed_ray_origins=transformed_ray_origins,
            transformed_ray_directions=transformed_ray_directions,
            white_back=white_back,
            last_back=last_back,
            return_aux_img=return_aux_img,
            idx_grad=idx_grad,
        )

        with torch.no_grad():
            inr_img_no_grad, aux_img_no_grad = self.points_forward(
                style_dict=style_dict,
                transformed_points=transformed_points,
                transformed_ray_directions_expanded=  # noqa
                transformed_ray_directions_expanded,
                num_steps=num_steps,
                hierarchical_sample=hierarchical_sample,
                z_vals=z_vals,
                clamp_mode=clamp_mode,
                nerf_noise=nerf_noise,
                transformed_ray_origins=transformed_ray_origins,
                transformed_ray_directions=transformed_ray_directions,
                white_back=white_back,
                last_back=last_back,
                return_aux_img=return_aux_img,
                idx_grad=idx_no_grad,
            )

        inr_img = scatter_points(
            idx_grad=idx_grad,
            points_grad=inr_img_grad,
            idx_no_grad=idx_no_grad,
            points_no_grad=inr_img_no_grad,
            num_points=num_points)

        inr_img = rearrange(inr_img, 'b (h w) c -> b c h w', h=img_size)
        inr_img = self.filters(inr_img)
        pitch_yaw = torch.cat([pitch, yaw], -1)

        if return_aux_img:
            aux_img = scatter_points(
                idx_grad=idx_grad,
                points_grad=aux_img_grad,
                idx_no_grad=idx_no_grad,
                points_no_grad=aux_img_no_grad,
                num_points=num_points)
            aux_img = rearrange(aux_img, 'b (h w) c -> b c h w', h=img_size)

            imgs = torch.cat([inr_img, aux_img])
            pitch_yaw = torch.cat([pitch_yaw, pitch_yaw])
        else:
            imgs = inr_img

        return imgs, pitch_yaw

    def points_forward(
        self,
        style_dict,
        transformed_points,
        transformed_ray_directions_expanded,
        num_steps,
        hierarchical_sample,
        z_vals,
        clamp_mode,
        nerf_noise,
        transformed_ray_origins,
        transformed_ray_directions,
        white_back,
        last_back,
        return_aux_img,
        idx_grad=None,
    ):
        """

        :param style_dict:
        :param transformed_points: (b, n, s, 3)
        :param transformed_ray_directions_expanded: (b, n, s, 3)
        :param num_steps: sampled points along a ray
        :param hierarchical_sample:
        :param z_vals: (b, n, s, 1)
        :param clamp_mode: 'relu'
        :param nerf_noise:
        :param transformed_ray_origins: (b, n, 3)
        :param transformed_ray_directions: (b, n, 3)
        :param white_back:
        :param last_back:
        :return:
        """
        device = transformed_points.device
        if idx_grad is not None:
            transformed_points = gather_points(
                points=transformed_points, idx_grad=idx_grad)
            transformed_ray_directions_expanded = gather_points(
                points=transformed_ray_directions_expanded, idx_grad=idx_grad)
            z_vals = gather_points(points=z_vals, idx_grad=idx_grad)
            transformed_ray_origins = gather_points(
                points=transformed_ray_origins, idx_grad=idx_grad)
            transformed_ray_directions = gather_points(
                points=transformed_ray_directions, idx_grad=idx_grad)

        transformed_points = rearrange(transformed_points,
                                       'b n s c -> b (n s) c')
        transformed_ray_directions_expanded = rearrange(
            transformed_ray_directions_expanded, 'b n s c -> b (n s) c')

        # Model prediction on course points
        coarse_output = self.siren(
            input=transformed_points,  # (b, n x s, 3)
            style_dict=style_dict,
            ray_directions=transformed_ray_directions_expanded,
        )
        coarse_output = rearrange(
            coarse_output, 'b (n s) rgb_sigma -> b n s rgb_sigma', s=num_steps)

        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            fine_points, fine_z_vals = self.get_fine_points_and_direction(
                coarse_output=coarse_output,
                z_vals=z_vals,
                dim_rgb=self.siren.rgb_dim,
                clamp_mode=clamp_mode,
                nerf_noise=nerf_noise,
                num_steps=num_steps,
                transformed_ray_origins=transformed_ray_origins,
                transformed_ray_directions=transformed_ray_directions)

            # Model prediction on re-sampled find points
            fine_output = self.siren(
                input=fine_points,  # (b, n x s, 3)
                style_dict=style_dict,
                # (b, n x s, 3)
                ray_directions=transformed_ray_directions_expanded,
            )
            fine_output = rearrange(
                fine_output,
                'b (n s) rgb_sigma -> b n s rgb_sigma',
                s=num_steps)

            # Combine course and fine points
            # (b, n, s, dim_rgb_sigma)
            all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
            all_z_vals = torch.cat([fine_z_vals, z_vals],
                                   dim=-2)  # (b, n, s, 1)
            _, indices = torch.sort(all_z_vals, dim=-2)  # (b, n, s, 1)
            all_z_vals = torch.gather(all_z_vals, -2, indices)  # (b, n, s, 1)
            # (b, n, s, dim_rgb_sigma)
            all_outputs = torch.gather(
                all_outputs, -2,
                indices.expand(-1, -1, -1, all_outputs.shape[-1]))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        # Create images with NeRF
        pixels_fea, depth, weights = fancy_integration(
            rgb_sigma=all_outputs,
            z_vals=all_z_vals,
            device=device,
            dim_rgb=self.siren.rgb_dim,
            white_back=white_back,
            last_back=last_back,
            clamp_mode=clamp_mode,
            noise_std=nerf_noise)

        inr_img = self.inr_net(pixels_fea, style_dict)

        if return_aux_img:
            # aux rgb_branch
            aux_img = self.aux_to_rbg(pixels_fea)
        else:
            aux_img = None

        return inr_img, aux_img

    def z_sampler(self, shape, device, dist='gaussian'):
        if dist == 'gaussian':
            z = torch.randn(shape, device=device)
        elif dist == 'uniform':
            z = torch.rand(shape, device=device) * 2 - 1
        return z

    def get_zs(self, b, batch_split=1):
        z_nerf = self.z_sampler(
            shape=(b, self.mapping_network_nerf.z_dim), device=self.device)
        z_inr = self.z_sampler(
            shape=(b, self.mapping_network_inr.z_dim), device=self.device)

        if batch_split > 1:
            zs_list = []
            z_nerf_list = z_nerf.split(b // batch_split)
            z_inr_list = z_inr.split(b // batch_split)
            for z_nerf_, z_inr_ in zip(z_nerf_list, z_inr_list):
                zs_ = {
                    'z_nerf': z_nerf_,
                    'z_inr': z_inr_,
                }
                zs_list.append(zs_)
            return zs_list
        else:
            zs = {
                'z_nerf': z_nerf,
                'z_inr': z_inr,
            }
            return zs

    def mapping_network(self, z_nerf, z_inr):
        style_dict = {}
        style_dict.update(self.mapping_network_nerf(z_nerf))
        style_dict.update(self.mapping_network_inr(z_inr))
        return style_dict

    def generate_avg_frequencies(self, num_samples=10000, device='cuda'):
        """Calculates average frequencies and phase shifts."""

        # z = torch.randn((num_samples, self.z_dim), device=device)
        zs = self.get_zs(num_samples)
        with torch.no_grad():
            style_dict = self.mapping_network(**zs)

        avg_styles = {}
        for name, style in style_dict.items():
            avg_styles[name] = style.mean(0, keepdim=True)

        self.avg_styles = avg_styles
        return avg_styles

    def staged_forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_device(self, device):
        # self.device = device
        # self.siren.device = device
        # self.generate_avg_frequencies()
        pass

    def forward_camera_pos_and_lookup(self,
                                      zs,
                                      img_size,
                                      fov,
                                      ray_start,
                                      ray_end,
                                      num_steps,
                                      h_stddev,
                                      v_stddev,
                                      h_mean,
                                      v_mean,
                                      hierarchical_sample,
                                      camera_pos,
                                      camera_lookup,
                                      psi=1,
                                      sample_dist=None,
                                      lock_view_dependence=False,
                                      clamp_mode='relu',
                                      nerf_noise=0.0,
                                      white_back=False,
                                      last_back=False,
                                      return_aux_img=False,
                                      grad_points=None,
                                      forward_points=None,
                                      up_vector=None,
                                      **kwargs):
        """Generates images from a noise vector, rendering parameters, and
        camera distribution. Uses the hierarchical sampling scheme described in
        NeRF.

        :param z: (b, z_dim)
        :param img_size:
        :param fov: face: 12
        :param ray_start: face: 0.88
        :param ray_end: face: 1.12
        :param num_steps: face: 12
        :param h_stddev: face: 0.3
        :param v_stddev: face: 0.155
        :param h_mean: face: pi/2
        :param v_mean: face: pi/2
        :param hierarchical_sample: face: true
        :param camera_pos: (b, 3)
        :param camera_lookup: (b, 3)
        :param psi: [0, 1]
        :param sample_dist: mode for sample_camera_positions, face: 'gaussian'
        :param lock_view_dependence: face: false
        :param clamp_mode: face: 'relu'
        :param nerf_noise:
        :param last_back: face: false
        :param white_back: face: false
        :param kwargs:
        :return:
        - pixels: (b, 3, h, w)
        - pitch_yaw: (b, 2)
        """

        # mapping network
        style_dict = self.mapping_network(**zs)

        if psi < 1:
            avg_styles = self.generate_avg_frequencies(device=self.device)
            style_dict = self.get_truncated_freq_phase(
                raw_style_dict=style_dict,
                avg_style_dict=avg_styles,
                raw_lambda=psi)

        if grad_points is not None and grad_points < img_size**2:
            imgs, pitch_yaw = self.part_grad_forward(
                style_dict=style_dict,
                img_size=img_size,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end,
                num_steps=num_steps,
                h_stddev=h_stddev,
                v_stddev=v_stddev,
                h_mean=h_mean,
                v_mean=v_mean,
                hierarchical_sample=hierarchical_sample,
                sample_dist=sample_dist,
                lock_view_dependence=lock_view_dependence,
                clamp_mode=clamp_mode,
                nerf_noise=nerf_noise,
                white_back=white_back,
                last_back=last_back,
                return_aux_img=return_aux_img,
                grad_points=grad_points,
                camera_pos=camera_pos,
                camera_lookup=camera_lookup,
            )
            return imgs, pitch_yaw
        else:
            imgs, pitch_yaw = self.whole_grad_forward(
                style_dict=style_dict,
                img_size=img_size,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end,
                num_steps=num_steps,
                h_stddev=h_stddev,
                v_stddev=v_stddev,
                h_mean=h_mean,
                v_mean=v_mean,
                hierarchical_sample=hierarchical_sample,
                sample_dist=sample_dist,
                lock_view_dependence=lock_view_dependence,
                clamp_mode=clamp_mode,
                nerf_noise=nerf_noise,
                white_back=white_back,
                last_back=last_back,
                return_aux_img=return_aux_img,
                forward_points=forward_points,
                camera_pos=camera_pos,
                camera_lookup=camera_lookup,
                up_vector=up_vector,
            )
            return imgs, pitch_yaw


class NeRFNetwork(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input
    points to -1, 1."""

    def __init__(self,
                 in_dim=3,
                 hidden_dim=256,
                 hidden_layers=2,
                 style_dim=512,
                 rgb_dim=3,
                 device=None,
                 name_prefix='nerf',
                 **kwargs):
        """

        :param z_dim:
        :param hidden_dim:
        :param rgb_dim:
        :param device:
        :param kwargs:
        """
        super().__init__()

        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.rgb_dim = rgb_dim
        self.style_dim = style_dim
        self.hidden_layers = hidden_layers
        self.name_prefix = name_prefix

        self.module_name_list = []

        self.style_dim_dict = {}

        self.network = nn.ModuleList()
        self.module_name_list.append('network')
        _out_dim = in_dim
        for idx in range(hidden_layers):
            _in_dim = _out_dim
            _out_dim = hidden_dim

            _layer = FiLMLayer(
                in_dim=_in_dim,
                out_dim=_out_dim,
                style_dim=style_dim,
                use_style_fc=True)

            self.network.append(_layer)
            self.style_dim_dict[f'{name_prefix}_w{idx}'] = _layer.style_dim

        self.final_layer = nn.Linear(hidden_dim, 1)
        # self.final_layer.apply(frequency_init(25))
        self.module_name_list.append('final_layer')

        _in_dim = hidden_dim
        _out_dim = hidden_dim // 2
        self.color_layer_sine = FiLMLayer(
            in_dim=_in_dim,
            out_dim=_out_dim,
            style_dim=style_dim,
            use_style_fc=True)
        self.style_dim_dict[
            f'{name_prefix}_rgb'] = self.color_layer_sine.style_dim
        self.module_name_list.append('color_layer_sine')

        self.color_layer_linear = nn.Sequential(nn.Linear(_out_dim, rgb_dim), )
        self.color_layer_linear.apply(kaiming_leaky_init)
        self.module_name_list.append('color_layer_linear')

        self.dim_styles = sum(self.style_dim_dict.values())

        # Don't worry about this, it was added to ensure compatibility with
        # another model. Shouldn't affect performance.
        self.gridwarper = UniformBoxWarp(0.24)

        models_dict = {}
        for name in self.module_name_list:
            models_dict[name] = getattr(self, name)
        models_dict['nerf'] = self

    def forward_with_frequencies_phase_shifts(self, input, style_dict,
                                              **kwargs):
        """

        :param input: (b, n, 3)
        :param style_dict:
        :param ray_directions:
        :param kwargs:
        :return:
        """

        input = self.gridwarper(input)
        x = input

        for index, layer in enumerate(self.network):
            style = style_dict[f'{self.name_prefix}_w{index}']

            x = layer(x, style)

        sigma = self.final_layer(x)

        # rgb branch
        style = style_dict[f'{self.name_prefix}_rgb']
        x = self.color_layer_sine(x, style)

        rbg = self.color_layer_linear(x)

        out = torch.cat([rbg, sigma], dim=-1)
        return out

    def forward(self, input, style_dict, ray_directions, **kwargs):
        """

        :param input: points xyz, (b, num_points, 3)
        :param style_dict:
        :param ray_directions: (b, num_points, 3)
        :param kwargs:
        :return:
        - out: (b, num_points, 4), rgb(3) + sigma(1)
        """

        out = self.forward_with_frequencies_phase_shifts(
            input=input,
            style_dict=style_dict,
            ray_directions=ray_directions,
            **kwargs)

        return out

    def get_freq_phase(self, style_dict, name):
        styles = style_dict[name]
        styles = rearrange(styles, 'b (n d) -> b d n', n=2)
        frequencies, phase_shifts = styles.unbind(-1)
        frequencies = frequencies * 15 + 30
        return frequencies, phase_shifts

    def staged_forward(self, transformed_points,
                       transformed_ray_directions_expanded, style_dict,
                       max_points, num_steps):

        batch_size, num_points, _ = transformed_points.shape

        rgb_sigma_output = torch.zeros(
            (batch_size, num_points, self.rgb_dim + 1), device=self.device)
        for b in range(batch_size):
            head = 0
            while head < num_points:
                tail = head + max_points
                rgb_sigma_output[b:b + 1, head:tail] = self(
                    input=transformed_points[b:b + 1,
                                             head:tail],  # (b, h x w x s, 3)
                    style_dict={
                        name: style[b:b + 1]
                        for name, style in style_dict.items()
                    },
                    ray_directions=transformed_ray_directions_expanded[
                        b:b + 1, head:tail])
                head += max_points
        rgb_sigma_output = rearrange(
            rgb_sigma_output,
            'b (hw s) rgb_sigma -> b hw s rgb_sigma',
            s=num_steps)
        return rgb_sigma_output


@MODULES.register_module()
class CIPSGenerator(nn.Module):

    def __init__(self,
                 input_dim,
                 style_dim,
                 hidden_dim=256,
                 pre_rgb_dim=32,
                 device=None,
                 name_prefix='inr',
                 **kwargs):
        """

        :param input_dim:
        :param style_dim:
        :param hidden_dim:
        :param pre_rgb_dim:
        :param device:
        :param name_prefix:
        :param kwargs:
        """
        super().__init__()

        self.device = device
        self.pre_rgb_dim = pre_rgb_dim
        self.name_prefix = name_prefix

        self.channels = {
            '4': hidden_dim,
            '8': hidden_dim,
            '16': hidden_dim,
            '32': hidden_dim,
            '64': hidden_dim,
            '128': hidden_dim,
            '256': hidden_dim,
            '512': hidden_dim,
            '1024': hidden_dim,
        }

        self.module_name_list = []

        self.style_dim_dict = {}

        _out_dim = input_dim

        network = OrderedDict()
        to_rbgs = OrderedDict()
        for i, (name, channel) in enumerate(self.channels.items()):
            _in_dim = _out_dim
            _out_dim = channel

            if name.startswith(('none', )):
                _linear_block = LinearBlock(
                    in_dim=_in_dim,
                    out_dim=_out_dim,
                    name_prefix=f'{name_prefix}_{name}')
                network[name] = _linear_block
            else:
                _film_block = SinBlock(
                    in_dim=_in_dim,
                    out_dim=_out_dim,
                    style_dim=style_dim,
                    name_prefix=f'{name_prefix}_w{name}')
                self.style_dim_dict.update(_film_block.style_dim_dict)
                network[name] = _film_block

            _to_rgb = ToRGB(
                in_dim=_out_dim, dim_rgb=pre_rgb_dim, use_equal_fc=False)
            to_rbgs[name] = _to_rgb

        self.network = nn.ModuleDict(network)
        self.to_rgbs = nn.ModuleDict(to_rbgs)
        self.to_rgbs.apply(frequency_init(100))
        self.module_name_list.append('network')
        self.module_name_list.append('to_rgbs')

        out_layers = []
        if pre_rgb_dim > 3:
            out_layers.append(nn.Linear(pre_rgb_dim, 3))
        out_layers.append(nn.Tanh())
        self.tanh = nn.Sequential(*out_layers)
        # self.tanh.apply(init_func.kaiming_leaky_init)
        self.tanh.apply(frequency_init(100))
        self.module_name_list.append('tanh')

        models_dict = {}
        for name in self.module_name_list:
            models_dict[name] = getattr(self, name)
        models_dict['cips'] = self

    def forward_orig(self, input, style_dict, img_size=1024, **kwargs):
        """

        :param input: points xyz, (b, num_points, 3)
        :param style_dict:
        :param ray_directions: (b, num_points, 3)
        :param kwargs:
        :return:
        - out: (b, num_points, 4), rgb(3) + sigma(1)
        """

        x = input
        img_size = str(2**int(np.log2(img_size)))

        rgb = 0
        for idx, (name, block) in enumerate(self.network.items()):
            # skip = int(name) >= 32
            if idx >= 4:
                skip = True
            else:
                skip = False
            x = block(x, style_dict, skip=skip)

            if idx >= 3:
                rgb = self.to_rgbs[name](x, skip=rgb)

            if name == img_size:
                break

        out = self.tanh(rgb)
        return out

    def forward(self,
                noise,
                style,
                return_noise=False,
                return_label=False,
                **kwargs):
        """This function implement a forward function in mmgen's style.

        Args:
            noise: The input feature.
            style: The style dict.

        Returns:
            output: Tensor shape as (bz, n_points, 4) or dict.
        """
        output = self.forward_orig(noise, style)

        if return_noise:
            output_dict = dict(
                fake_pixels=output, style=style, noise_batch=noise)
            return output_dict
        return output
