_base_ = [
    '../_base_/datasets/unconditional_imgs_128x128_pil_backend.py',
    '../_base_/models/graf/carla.py', '../_base_/default_runtime.py'
]

model = dict(camera=dict(H_range=[0, 128], W_range=[0, 128]))

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
        interval=5000,
        rerange=False,
        kwargs=dict(sample_model='ema')),
]

inception_pkl = './work_dirs/inception_pkl/carla_128.pkl'
evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=[
        dict(
            type='FID',
            num_images=1000,
            inception_pkl=inception_pkl,
            bgr2rgb=True,
            inception_args=dict(type='StyleGAN')),
        # dict(type='KID', num_images=2000)
    ],
    best_metric='fid',
    sample_kwargs=dict(sample_model='ema'))

metrics = dict(
    fid1k=dict(
        type='FID',
        num_images=1000,
        inception_pkl=inception_pkl,
        bgr2rgb=True,
        inception_args=dict(type='StyleGAN')),
    # kid2k=dict(type='KID'),
)

total_iters = 640000
imgs_root = './data/carla'
data = dict(
    samples_per_gpu=8,
    train=dict(imgs_root=imgs_root),
    val=dict(imgs_root=imgs_root))
