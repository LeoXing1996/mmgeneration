_base_ = [
    '../_base_/datasets/blender_dataset.py',
    '../_base_/models/nerf/blender.py', '../_base_/default_runtime.py'
]

ray_sampler = dict(type='PrecropSampler', precrop_frac=0.5, n_points=1024)

camera = dict(
    camera_angle_x=0.6911112070083618,
    H_range=[0, 800],
    W_range=[0, 800],
    near=2.0,
    far=6.0,
    which_hand='left',
    ray_sampler=ray_sampler)

model = dict(camera=camera)

custom_hooks = [
    dict(
        type='VisualizeReconstructionSamples',
        output_dir='training_samples',
        interval=5000,
        num_samples=-1,
        bgr2rgb=True,
        rerange=False,
        vis_keys=['rgb_final', 'rgb_coarse', 'real_pixels']),
    dict(type='PrecropRaySamplerHook', precrop_frac_iter=500)
]

evaluation = dict(
    type='GenerativeEvalHook',
    interval=5000,
    metrics=dict(
        type='PSNR',
        num_images=-1,
        data_info=dict(reals='real_pixels', fakes='rgb_final'),
        input_order='NH*WC',
        input_scale=[0, 1]),
    best_metric='psnr',
    sample_kwargs=dict(sample_model='orig'))

dataroot = './data/blender/lego'
data = dict(
    train=dict(dataset=dict(dataroot=dataroot)),
    val=dict(dataroot=dataroot, skip_per_image=50),
    test=dict(dataroot=dataroot))

metrics = dict(
    psnr=dict(
        type='PSNR',
        num_images=-1,
        data_info=dict(reals='real_pixels', fakes='rgb_final'),
        input_order='NH*WC',
        input_scale=[0, 1]))
