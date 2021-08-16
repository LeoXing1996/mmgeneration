# this config follow the setting `launch_SAGAN_bz128x2_ema.sh` from BigGAN repo
# 1. use eps=1e-8 for Adam (seems same as original eps)
# 2. use eps=1e-8 in SN
# 3. w/o SyncBN
# 4. ema start from 2000 step
# 5. total bz=128 on 4 Titan XP (4x32) and use step accu=2

# add the frequency for log and vis to inspect what cause collapse
_base_ = [
    '../_base_/models/sagan_128x128.py',
    '../_base_/datasets/imagenet_128_memcache.py',
    '../_base_/default_runtime.py'
]

init_cfg = dict(type='BigGAN')
model = dict(
    generator=dict(
        num_classes=1000,
        init_cfg=init_cfg,
        norm_eps=1e-5,
        sn_eps=1e-8,
        auto_sync_bn=False,
        with_embedding_spectral_norm=False),
    discriminator=dict(num_classes=1000, init_cfg=init_cfg, sn_eps=1e-8),
)

disc_step = 1
n_accu = 1
n_disc = disc_step * n_accu

train_cfg = dict(
    disc_steps=n_disc, batch_accumulation_steps=n_accu, use_ema=True)

lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=2)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=1000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=n_disc,
        start_iter=2000 * n_disc,
        interp_cfg=dict(momentum=0.9999, momentum_nontrainable=0.999),
        update_sn=True,
        priority='VERY_HIGH')
]

inception_pkl = './work_dirs/inception_pkl/imagenet.pkl'

evaluation = dict(
    type='GenerativeEvalHook',
    interval=dict(
        milestones=[800000 * n_disc], interval=[10000 * n_disc,
                                                2000 * n_disc]),
    metrics=[
        dict(
            type='FID',
            num_images=50000,
            inception_pkl=inception_pkl,
            bgr2rgb=True,
            inception_args=dict(type='StyleGAN')),
        dict(type='IS', num_images=50000)
    ],
    best_metric=['fid', 'is'],
    sample_kwargs=dict(sample_model='ema'))

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='PaviLoggerHook',
            init_kwargs=dict(project='sagan-imagenet-mmgen')),
    ])

total_iters = 1000000 * n_disc

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl=inception_pkl,
        inception_args=dict(type='StyleGAN')),
    IS50k=dict(type='IS', num_images=50000))

optimizer = dict(
    generator=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-8),
    discriminator=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-8))

# train on 4 gpus
data = dict(samples_per_gpu=16)
