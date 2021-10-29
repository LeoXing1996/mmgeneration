_base_ = [
    '../_base_/models/ddpm_32x32.py', '../_base_/datasets/cifar10_noaug.py',
    '../_base_/default_runtime.py'
]

data = dict(samples_per_gpu=8)

lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=10000)
]

evaluation = None

# total_iters = 100000
total_iters = 10
# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

metric = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl=None,
        inception_args=dict(type='StyleGAN')))
