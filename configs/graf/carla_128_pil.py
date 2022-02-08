_base_ = [
    '../_base_/datasets/unconditional_imgs_128x128_pil_backend.py',
    '../_base_/models/graf/carla.py', '../_base_/default_runtime.py'
]

ray_sampler = dict(
    type='FlexGridRaySampler',
    min_scale=0.25,
    max_scale=1,
    random_shift=False,
    random_scale=False,
    n_points=1024)

# v_dist: 85 deg, convert to degrees via arccos(1 - 2 * v) * 180. / pi
camera = dict(
    type='RandomPoseCamera',
    fov=30,
    H_range=[0, 128],
    W_range=[0, 128],
    near=7.5,
    far=12.5,
    which_hand='left',
    camera_sample_mode='spherical',
    u_dist=[0, 1],
    v_dist=[0, 0.45642212862617093],
    radius_dist=10,
    ray_sampler=ray_sampler,
    degree2radian=True)

model = dict(camera=camera)

log_config = dict(
    interval=10, hooks=[
        dict(type='TextLoggerHook'),
    ])

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=1,
        start_iter=0,
        interp_cfg=dict(momentum=0.999),
        priority='VERY_HIGH'),
    dict(type='FlexGridRaySamplerHook', scale_annel=0.0025),
    dict(
        type='GRAFVisHook',
        output_dir='training_samples',
        nrow=2,
        num_samples=2,
        interval=100,
        rerange=False,
        kwargs=dict(sample_model='ema')),
    # upload ckpts
    # dict(type='PetrelUploadHook', rm_orig=False),
    # upload imgs
    # dict(
    #     type='PetrelUploadHook',
    #     rm_orig=False,
    #     data_path='training_samples',
    #     suffix='.png')
]

inception_pkl = './work_dirs/inception_pkl/carla_128_pil.pkl'
evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=[
        dict(
            type='FID',
            num_images=2000,
            inception_pkl=inception_pkl,
            bgr2rgb=False,
            inception_args=dict(type='StyleGAN')),
        # dict(type='KID', num_images=2000)
    ],
    best_metric='fid',
    sample_kwargs=dict(sample_model='ema'))

metrics = dict(
    fid2k=dict(
        type='FID',
        num_images=1000,
        inception_pkl=inception_pkl,
        bgr2rgb=True,  # TODO: to debug official ckpts
        inception_args=dict(type='StyleGAN')),
    # kid2k=dict(type='KID'),
)

total_iters = 640000
imgs_root = './data/carla'
data = dict(
    samples_per_gpu=8,
    train=dict(imgs_root=imgs_root),
    val=dict(imgs_root=imgs_root))
