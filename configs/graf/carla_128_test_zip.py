_base_ = [
    '../_base_/models/graf/carla.py', '../_base_/default_runtime.py',
    '../../c128_zip.py'
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
        kwargs=dict(sample_model='ema'))
]

total_iters = 100000
