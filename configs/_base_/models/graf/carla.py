ray_sampler = dict(
    type='FlexGridRaySampler',
    min_scale=0.25,
    max_scale=1,
    random_shift=False,
    random_scale=False,
    n_points=1024)

# v_dist: 85 deg, convert to degrees via arccos(1 - 2 * v) * 180. / pi
camera = dict(
    type='RandomPoseCamera',
    fov=30,
    H_range=[0, 128],
    W_range=[0, 128],
    near=7.5,
    far=12.5,
    which_hand='left',
    camera_sample_mode='spherical',
    u_dist=[0, 1],
    v_dist=[0, 0.45642212862617093],
    radius_dist=10,
    ray_sampler=ray_sampler,
    degree2radian=True)

# to save memory, this config use inplace-relu
model = dict(
    type='GRAF',
    camera=camera,
    generator=dict(
        type='GRAFGenerator',
        shape_dim=128,
        app_dim=128,
        use_inplace_act=True,
        alpha_act_cfg=dict(
            type='NoisyReLU', noise_std=1, inplace=True, end_iteration=4000),
        pose_embedding=dict(n_freq=4, include_input=True, ignore_pi=True),
        points_embedding=dict(n_freq=10, include_input=True, ignore_pi=True)),
    discriminator=dict(
        type='GRAFDiscriminator',
        base_channels=64,
        input_channels=3,
        inplace_relu=True),
    gan_loss=dict(type='GANLoss', gan_type='vanilla'),
    disc_auxiliary_loss=dict(
        type='R1GradientPenalty',
        norm_mode='HWC',
        loss_weight=10,
        data_info=dict(real_data='real_imgs', discriminator='disc')),
    white_background=True,
    num_samples_per_ray=64)

train_cfg = dict(noise_cfg='uniform', use_ema=True)
test_cfg = None

lr_config = dict(
    policy='Step', by_epoch=False, step=[50000, 100000, 200000], gamma=0.1)

optimizer = dict(
    generator=dict(type='RMSprop', lr=5e-4, alpha=0.99, eps=1e-8),
    discriminator=dict(type='RMSprop', lr=1e-4, alpha=0.99, eps=1e-8))
