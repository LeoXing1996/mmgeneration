_base_ = ['../_base_/datasets/blender_dataset_dist.py', './lego.py']

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='PaviLoggerHook', init_kwargs=dict(project='NeRF'))
    ])
