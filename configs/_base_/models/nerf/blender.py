model = dict(
    type='NeRF',
    neural_render=dict(
        type='NeRFRenderer',
        pose_embedding=dict(n_freq=4, include_input=True, ignore_pi=True),
        points_embedding=dict(n_freq=10, include_input=True, ignore_pi=True)),
    neural_render_fine=dict(
        type='NeRFRenderer',
        pose_embedding=dict(n_freq=4, include_input=True, ignore_pi=True),
        points_embedding=dict(n_freq=10, include_input=True, ignore_pi=True)),
    hierarchical_sampling=True,
    n_importance=128,
    white_background=True,
    num_samples_per_ray=64,
    nerf_loss=[
        dict(
            type='MSELoss',
            data_info=dict(pred='rgb_final', target='real_pixels'),
            loss_name='loss_fine'),
        dict(
            type='MSELoss',
            data_info=dict(pred='rgb_coarse', target='real_pixels'),
            loss_name='loss_coarse'),
    ])

train_cfg = dict(noise_cfg='uniform')
test_cfg = None

lr_config = dict(policy='TFStep', by_epoch=False, step=500 * 1000, gamma=0.1)

optimizer = dict(type='Adam', lr=5e-4, betas=(0.5, 0.999))

total_iters = 500000
