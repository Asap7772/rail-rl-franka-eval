import os.path as osp
import multiworld.envs.mujoco as mwmj
import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.torch.vae.conv_vae import imsize48_default_architecture
from railrl.torch.grill.ray_experiments import grill_her_twin_sac_online_vae_full_experiment
import ray.tune as tune

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        env_id='SawyerDoorHookResetFreeEnv-v1',
        init_camera=tune.function(sawyer_door_env_camera_v0),
        grill_variant=dict(
            save_video=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=20,
                lr=1e-3,
            ),
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            twin_sac_trainer_kwargs=dict(
                reward_scale=1,
                discount=0.99,
                soft_target_tau=1e-3,
                target_update_period=1,
                use_automatic_entropy_tuning=True,
            ),
            max_path_length=100,
            # max_path_length=5,
            algo_kwargs=dict(
                batch_size=256,
                num_epochs=170,
                num_eval_steps_per_epoch=1000,
                num_expl_steps_per_train_loop=500,
                num_trains_per_train_loop=1000,
                min_num_steps_before_training=2000,
                # num_epochs=5,
                # num_eval_steps_per_epoch=5,
                # num_expl_steps_per_train_loop=5,
                # num_trains_per_train_loop=10,
                # min_num_steps_before_training=10,
                vae_training_schedule=tune.function(vae_schedules.custom_schedule),
                oracle_data=False,
                vae_save_period=50,
                parallel_vae_train=False,
            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(100000),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    num_latents_to_sample=10,
                ),
                power=-0.5,
                relabeling_goal_sampling_mode='custom_goal_sampler',
            ),
            exploration_goal_sampling_mode='custom_goal_sampler',
            evaluation_goal_sampling_mode='presampled',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            presampled_goals_path=osp.join(
                osp.dirname(mwmj.__file__),
                "goals",
                "door_goals.npy",
            ),
            presample_goals=True,
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            algorithm='Skew-Fit-SAC',
        ),
        train_vae_variant=dict(
            representation_size=16,
            beta=20,
            num_epochs=0,
            dump_skew_debug_plots=False,
            decoder_activation='gaussian',
            generate_vae_dataset_kwargs=dict(
                N=2,
                test_p=.9,
                use_cached=True,
                show=False,
                oracle_dataset=False,
                n_random_steps=1,
                non_presampled_goal_img_is_garbage=True,
            ),
            vae_kwargs=dict(
                decoder_distribution='gaussian_identity_variance',
                input_channels=3,
                architecture=imsize48_default_architecture,
            ),
            algo_kwargs=dict(
                lr=1e-3,
                use_parallel_dataloading=False,
            ),
            save_period=1,
        ),
    )

    from railrl.launchers.ray.launcher import launch_experiment
    n_seeds = 6
    mode = 'gcp'
    exp_prefix = 'skew-fit-door-from-gcp-ray-7'

    launch_experiment(
        mode=mode,
        use_gpu=True,
        local_launch_variant=dict(
            seeds=n_seeds,
            init_algo_functions_and_log_fnames=grill_her_twin_sac_online_vae_full_experiment(),
            exp_variant=variant,
            checkpoint_freq=50,
            exp_prefix=exp_prefix,
            local_ray=False,
            resources_per_trial={
                'cpu': 2,
                'gpu': 1/3,
            }
        ),
        remote_launch_variant=dict(
            # head_instance_type='m1.xlarge',
            # max_spot_price=.2,
        ),
        docker_variant=dict(),
        cluster_name=exp_prefix
    )

