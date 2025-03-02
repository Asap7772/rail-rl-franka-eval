import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v3
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import grill_her_td3_full_experiment
from railrl.torch.vae.dataset.generate_goal_dataset import generate_goal_dataset_using_set_to_goal

if __name__ == "__main__":
    variant = dict(
        imsize=84,
        init_camera=sawyer_pusher_camera_upright_v3,
        grill_variant=dict(
            save_video=True,
            save_video_period=50,
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
                    num_updates_per_env_step=4,
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
                max_size=int(1e6),
                fraction_goals_are_rollout_goals=0,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='OFFLINE-AE-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.8,
            exploration_type='ou',
            training_mode='test',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            generate_goal_dataset_fctn=generate_goal_dataset_using_set_to_goal,
            goal_generation_kwargs=dict(
                num_goals=5000,
                use_cached_dataset=False,
                show=False,
            ),
            presample_goals=True,
            presampled_goals_path='goals/goals_n5000_VAEWrappedEnv(ImageEnv(<SawyerPushAndReachXYEnv<SawyerPushAndReachXYEnv-No-Arena-v1>>)).npy',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            )
        ),
        train_vae_variant=dict(
            vae_path=None,
            representation_size=16,
            beta=0,
            num_epochs=2500,
            dump_skew_debug_plots=False,
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=5000,
                oracle_dataset=True,
                use_cached=False,
                vae_dataset_specific_kwargs=dict(),
                show=False,
                dataset_path='datasets/SawyerPushAndReachXYEnv-No-Arena-v1_N5000_sawyer_pusher_camera_upright_v3_imsize84_random_oracle_split_0.npy',
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=True,
                batch_size=64,
                lr=1e-3,
            ),
            save_period=50,
        ),
    )

    search_space = {
        'env_id': ['SawyerPushAndReachXYEnv-No-Arena-v0', 'SawyerPushAndReachXYEnv-No-Arena-v1']
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'sawyer_pusher_offline_ae_final'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if variant['env_id'] == 'SawyerPushAndReachXYEnv-No-Arena-v0':
            variant['train_vae_variant']['generate_vae_dataset_kwargs']['dataset_path'] = \
                'datasets/SawyerPushAndReachXYEnv-No-Arena-v0_N5000_sawyer_pusher_camera_upright_v3_imsize84_random_oracle_split_0.npy'
            variant['grill_variant']['presampled_goals_path']= 'goals/goals_n5000_VAEWrappedEnv(ImageEnv(<SawyerPushAndReachXYEnv<SawyerPushAndReachXYEnv-No-Arena-v0>>)).npy'
        else:
            variant['train_vae_variant']['generate_vae_dataset_kwargs']['dataset_path'] = \
                'datasets/SawyerPushAndReachXYEnv-No-Arena-v1_N5000_sawyer_pusher_camera_upright_v3_imsize84_random_oracle_split_0.npy'
            variant['grill_variant'][
                'presampled_goals_path'] = 'goals/goals_n5000_VAEWrappedEnv(ImageEnv(<SawyerPushAndReachXYEnv<SawyerPushAndReachXYEnv-No-Arena-v1>>)).npy'
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
          )
