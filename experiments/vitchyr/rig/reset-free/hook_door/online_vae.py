import railrl.misc.hyperparameter as hyp
from railrl.torch.vae.dataset.generate_goal_dataset import generate_goal_dataset_using_policy
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v3
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import grill_her_td3_online_vae_full_experiment
import railrl.torch.vae.vae_schedules as vae_schedules

if __name__ == "__main__":
    variant = dict(
        double_algo=False,
        imsize=48,
        env_class=SawyerDoorHookEnv,
        init_camera=sawyer_door_env_camera_v3,
        env_kwargs=dict(
            # goal_low=(-0.1, 0.525, 0.05, 0),
            # goal_high=(0.0, 0.65, .075, 0.523599),
            # hand_low=(-0.1, 0.525, 0.05),
            # hand_high=(0., 0.65, .075),
            # max_angle=0.523599,
            # xml_path='sawyer_xyz/sawyer_door_pull_hook_30.xml',

            goal_low=(-0.1, 0.42, 0.05, 0),
            goal_high=(0.0, 0.65, .075, 1.0472),
            hand_low=(-0.1, 0.42, 0.05),
            hand_high=(0., 0.65, .075),
            max_angle=1.0472,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
        ),
        grill_variant=dict(
            save_video=False,
            online_vae_beta=2.5,
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=350,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=500,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=2,
                    collection_mode='online-parallel',
                    # collection_mode='online',
                    reward_scale=1,
                    parallel_env_params=dict(
                        num_workers=1,
                    )
                ),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
                online_vae_kwargs=dict(
                   vae_training_schedule=vae_schedules.every_six,
                    oracle_data=False,
                    vae_save_period=25,
                    parallel_vae_train=False,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(100000),
                fraction_goals_are_rollout_goals=0.2,
                fraction_resampled_goals_are_env_goals=0.5,
                exploration_rewards_scale=1.0,
                exploration_rewards_type='reconstruction_error',
                vae_priority_type='None',
                alpha=1,
            ),
            algorithm='ONLINE-VAE-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.8,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            generate_goal_dataset_fctn=generate_goal_dataset_using_policy,
            goal_generation_kwargs=dict(
                num_goals=1000,
                use_cached_dataset=False,
                policy_file='manual-upload/new-door-policy-from-murtaza.pkl',
                path_length=100,
                show=False,
                save_filename='/tmp/hook_door_n1000.npy',
            ),
            presampled_goals_path='manual-upload/hook_door_n1000.npy',
            presample_goals=True,
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            )
        ),
        train_vae_variant=dict(
            representation_size=16,
            beta=1.0,
            num_epochs=0,
            generate_vae_dataset_kwargs=dict(
                N=100,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                n_random_steps=1,
                non_presampled_goal_img_is_garbage=True,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                lr=1e-3,
            ),
            save_period=5,
        ),
    )

    search_space = {
        'env_kwargs.reset_free': [
            True,
        ],
        'train_vae_variant.algo_kwargs.gaussian_decoder_loss': [
            # True,
            False,
        ],
        'grill_variant.replay_buffer_kwargs.exploration_rewards_type': [
            'reconstruction_error',
            # 'None',
        ],
        'grill_variant.replay_buffer_kwargs.exploration_rewards_scale': [
            0, 1, 0.1,
        ],
        'grill_variant.replay_buffer_kwargs.vae_priority_type': [
            # 'bernoulli_inv_prob',
            # 'gaussian_inv_prob',
            # 'reconstruction_error',
            'None',
        ],
        'grill_variant.replay_buffer_kwargs.power': [
            0
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    # n_seeds = 5
    mode = 'slurm_singularity'
    # exp_prefix = 'sawyer-hook-settings-from-murtaza-expl-rew-sweep'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_td3_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
                time_in_mins=10*60,
                verbose=True,
          )
