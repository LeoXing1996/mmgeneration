# in SAGAN's official tensorflow implementation, author use tfrecord extracted
# by script from improved-gan, which use center-crop in train pipeline, refers
# to https://github.com/openai/improved-gan/blob/master/imagenet/convert_imagenet_to_records.py  # noqa

# To be noted that, this config should be trained with 4 gpus.

_base_ = [
    '../_base_/models/sagan_128x128.py', '../_base_/datasets/imagenet_128.py'
    '../_base_/default_runtime.py'
]

init_cfg = dict(type='SAGAN')
model = dict(
    generator=dict(num_classes=1000, init_cfg=init_cfg, auto_sync_bn=False),
    discriminator=dict(num_classes=1000, init_cfg=init_cfg))

lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)
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
    interval=10000,
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

n_disc = 1
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
    generator=dict(type='Adam', lr=0.0001, betas=(0.0, 0.9)),
    discriminator=dict(type='Adam', lr=0.0004, betas=(0.0, 0.9)))

data = dict(samples_per_gpu=64)
