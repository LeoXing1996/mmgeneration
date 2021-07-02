# follow the setting in chainer version
# features:
#   1. use learning rate decay (start from 400000 (* 5))
#   2. use xavier init and w/o reweight embedding
#   3. betas in Adam set as (0, 0.9)
#   4. only train 450000 (* 5) steps
#   5. use random noise augmentation

_base_ = [
    '../_base_/models/sngan_proj_128x128.py',
    '../_base_/datasets/imagenet_128.py',
    '../_base_/default_runtime.py',
]

num_classes = 1000
model = dict(
    generator=dict(num_classes=num_classes, init_cfg=dict(type='sngan')),
    discriminator=dict(num_classes=num_classes, init_cfg=dict(type='sngan')))

n_disc = 5
lr_config = dict(
    policy='Linear',
    target_lr=0,
    start=400000 * n_disc,
    interval=n_disc,
    by_epoch=False)

checkpoint_config = dict(interval=50000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=1000)
]

inception_pkl = \
    './work_dirs/inception_pkl/imagenet.pkl'

evaluation = dict(
    type='GenerativeEvalHook',
    interval=50000,
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
    sample_kwargs=dict(sample_model='orig'))

total_iters = 450000 * n_disc
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
    generator=dict(type='Adam', lr=0.0002, betas=(0.0, 0.9)),
    discriminator=dict(type='Adam', lr=0.0002, betas=(0.0, 0.9)))

data = dict(samples_per_gpu=16)
