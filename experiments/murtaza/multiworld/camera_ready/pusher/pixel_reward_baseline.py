import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v3
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import HER_baseline_her_td3_full_experiment

if __name__ == "__main__":
    variant = dict(
        imsize=84,
        init_camera=sawyer_pusher_camera_upright_v3,
        grill_variant=dict(
            save_video=False,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=505,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=1,
                    collection_mode='online',
                    parallel_env_params=dict(
                        num_workers=1,
                    ),
                    reward_scale=1,
                ),
                her_kwargs=dict(),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e4),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0,
            ),
            algorithm='PIX-REWARD-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.8,
            exploration_type='ou',
            training_mode='test',
            testing_mode='test',
            observation_key='image_observation',
            desired_goal_key='image_desired_goal',
            goal_generation_kwargs=dict(
                num_goals=5000,
                use_cached_dataset=False,
                show=False,
            ),
            presample_goals=True,
            presampled_goals_path='goals/goals_n5000_VAEWrappedEnv(ImageEnv(<SawyerPushAndReachXYEnv<SawyerPushAndReachXYEnv-No-Arena-v1>>)).npy',
            cnn_params=dict(
                kernel_sizes=[5, 5, 5],
                n_channels=[16, 32, 32],
                strides=[3, 3, 3],
                pool_sizes=[1, 1, 1],
                hidden_sizes=[32, 32],
                paddings=[0, 0, 0],
                use_batch_norm=False,
            ),
        ),
        train_vae_variant=dict(
            vae_path=None,
            representation_size=16,
            beta=.5,
            num_epochs=1000,
            dump_skew_debug_plots=False,
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=5000,
                oracle_dataset=False,
                use_cached=False,
                oracle_dataset_from_policy=True,
                non_presampled_goal_img_is_garbage=True,
                vae_dataset_specific_kwargs=dict(),
                policy_file='10-23-sawyer-door-v5-es-sweep/10-23-sawyer_door_v5_es_sweep_2018_10_24_00_13_10_id000--s3382/params.pkl',
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
            ),
            save_period=10,
        ),
    )

    search_space = {
        'env_id': ['SawyerPushAndReachXYEnv-No-Arena-v0', 'SawyerPushAndReachXYEnv-No-Arena-v1'],
        'grill_variant.algo_kwargs.base_kwargs.num_updates_per_env_step': [1, 4]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'sawyer_pusher_pix_reward_baseline_final'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if variant['env_id'] == 'SawyerPushAndReachXYEnv-No-Arena-v0':
            variant['train_vae_variant']['generate_vae_dataset_kwargs']['dataset_path'] = \
                'datasets/SawyerPushAndReachXYEnv-No-Arena-v0_N5000_sawyer_pusher_camera_upright_v3_imsize84_random_oracle_split_0.npy'
            variant['grill_variant'][
                'presampled_goals_path'] = 'goals/goals_n5000_VAEWrappedEnv(ImageEnv(<SawyerPushAndReachXYEnv<SawyerPushAndReachXYEnv-No-Arena-v0>>)).npy'
        else:
            variant['train_vae_variant']['generate_vae_dataset_kwargs']['dataset_path'] = \
                'datasets/SawyerPushAndReachXYEnv-No-Arena-v1_N5000_sawyer_pusher_camera_upright_v3_imsize84_random_oracle_split_0.npy'
            variant['grill_variant'][
                'presampled_goals_path'] = 'goals/goals_n5000_VAEWrappedEnv(ImageEnv(<SawyerPushAndReachXYEnv<SawyerPushAndReachXYEnv-No-Arena-v1>>)).npy'
        for _ in range(n_seeds):
            run_experiment(
                HER_baseline_her_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
          )
