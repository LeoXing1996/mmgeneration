# MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"  # noqa
# DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"  # noqa
# TRAIN_FLAGS="--lr 1e-4 --batch_size 128"  # noqa

model = dict(
    type='BasicGaussianDiffusion',
    num_timesteps=1000,
    betas_cfg=dict(type='linear'),
    denoising=dict(
        type='DenoisingUnet',
        image_size=256,
        in_channels=3,
        base_channels=128,
        resblocks_per_downsample=2,
        attention_res=[16],
        use_scale_shift_norm=False,
        dropout=0,
        num_heads=1,
        use_rescale_timesteps=False,
        output_cfg=dict(mean='eps', var='learned_range'),
        resblock_cfg=dict(type='DDPM_ResBlock')),
    timestep_sampler=dict(type='UniformTimeStepSampler'),
    ddpm_loss=[
        dict(
            type='DDPMVLBLoss',
            data_info=dict(
                mean_pred='mean_pred',
                mean_target='mean_posterior',
                logvar_pred='logvar_pred',
                logvar_target='logvar_posterior'),
            log_cfgs=[
                dict(
                    type='quartile',
                    prefix_name='loss_vlb',
                    total_timesteps=1000),
                dict(type='name')
            ]),
        dict(
            type='DDPMMSELoss',
            log_cfgs=dict(
                type='quartile', prefix_name='loss_mse', total_timesteps=1000),
        )
    ],
)

train_cfg = dict(use_ema=True, real_img_key='img')
test_cfg = None
optimizer = dict(denoising=dict(type='AdamW', lr=1e-4, weight_decay=0))
