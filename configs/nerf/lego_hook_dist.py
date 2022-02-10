_base_ = ['./lego_hook.py']

pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        io_backend='memcached',
        server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
        client_cfg='/mnt/lustre/share/memcached_client/client.conf',
        sys_path='/mnt/lustre/share/pymc/py3',
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

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='PaviLoggerHook', init_kwargs=dict(project='NeRF'))
    ])
