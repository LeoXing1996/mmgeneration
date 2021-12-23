dataset_type = 'BlenderDataset'

pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        io_backend='disk',
        flag='unchanged'),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[0.] * 4,
        std=[255.] * 4,
        to_rgb=False),
    dict(type='BackGroundtoWhite', keys=['real_img'], white_background=True),
    dict(type='ToTensor', keys=['transform_matrix']),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(
        type='Collect',
        keys=['real_img', 'transform_matrix'],
        meta_keys=['real_img_path'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=5,
    persistent_workers=True,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(type=dataset_type, pipeline=pipeline, split='train')),
    val=dict(type=dataset_type, pipeline=pipeline, split='val'),
    test=dict(type=dataset_type, pipeline=pipeline, split='test'))
