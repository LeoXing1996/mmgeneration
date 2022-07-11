dataset_type = 'UnconditionalImageDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        io_backend='disk',
    ),
    dict(type='Resize', scale=(512, 384)),
    dict(
        type='NumpyPad',
        padding=((64, 64), (0, 0), (0, 0)),
    ),
    dict(type='Flip', keys=['img'], direction='horizontal'),
    dict(type='PackGenInputs', keys=['img'], meta_keys=['img_path'])
]

val_pipeline = train_pipeline

# `batch_size` and `data_root` need to be set.
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
