import railrl.misc.hyperparameter as hyp
from experiments.murtaza.multiworld.decoder_distributions.mse_decoder import architecture
from railrl.torch.vae.conv_vae import ConvVAE
from railrl.torch.vae.dataset.generate_goal_dataset import generate_goal_dataset_using_policy
from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v0
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import grill_her_twin_sac_online_vae_full_experiment
import railrl.torch.vae.vae_schedules as vae_schedules

if __name__ == "__main__":
    variant=dict(
    double_algo = False,
    online_vae_exploration = False,
    imsize = 48,
    env_id = 'SawyerDoorHookResetFreeEnv-v0',
    init_camera = sawyer_door_env_camera_v0,
    grill_variant = dict(
        save_video=True,
        online_vae_beta=2.5,
        save_video_period=50,
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=1010,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                min_num_steps_before_training=10000,
                batch_size=128,
                max_path_length=100,
                discount=0.99,
                num_updates_per_env_step=2,
                collection_mode='online-parallel',
                parallel_env_params=dict(
                    num_workers=1,
                ),
                reward_scale=1,
            ),
            her_kwargs=dict(
            ),
            twin_sac_kwargs=dict(
                train_policy_with_reparameterization=True,
                soft_target_tau=1e-3,  # 1e-2
                policy_update_period=1,
                target_update_period=1,  # 1
                use_automatic_entropy_tuning=True,
            ),
            online_vae_kwargs=dict(
                vae_training_schedule=vae_schedules.every_other,
                oracle_data=False,
                vae_save_period=50,
                parallel_vae_train=False,
            ),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(100000),
            fraction_goals_are_rollout_goals=0,
            fraction_resampled_goals_are_env_goals=0.5,
            exploration_rewards_type='None',
            vae_priority_type='image_gaussian_inv_prob',
            power=1,
        ),
        normalize=False,
        render=False,
        exploration_noise=0,
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
            use_cached_dataset=True,
            policy_file='data/doodads3/11-09-her-twin-sac-door/11-09-her-twin-sac-door_2018_11_10_02_17_10_id000--s16215/params.pkl',
            path_length=100,
            show=False,
            tag='_twin_sac'
        ),
        presampled_goals_path='goals/SawyerDoorHookResetFreeEnv-v0_N1000_imsize48goals_twin_sac.npy',
        presample_goals=True,
        vae_wrapped_env_kwargs=dict(
            sample_from_true_prior=True,
        ),
        algorithm='ONLINE-VAE-SAC-GAUSSIAN-HER-TD3',
        ),
        train_vae_variant=dict(
            representation_size=16,
            beta=1.0,
            num_epochs=0,
            dump_skew_debug_plots=False,
            generate_vae_dataset_kwargs=dict(
                N=100,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                n_random_steps=1,
                non_presampled_goal_img_is_garbage=True,
                tag='_twin_sac'
            ),
            vae_class=ConvVAE,
            vae_kwargs=dict(
                input_channels=3,
                architecture=architecture,
                decoder_distribution='gaussian_identity_variance',
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
        'grill_variant.algo_kwargs.online_vae_kwargs.vae_training_schedule':[vae_schedules.every_other, vae_schedules.every_six],
        'grill_variant.online_vae_beta': [.5, 1, 2.5],
        'grill_variant.replay_buffer_kwargs.vae_priority_type':['None', 'image_gaussian_inv_prob'],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 2
    mode = 'ec2'
    exp_prefix = 'door_online_vae_inv_gaussian_priority'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                grill_her_twin_sac_online_vae_full_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
          )
