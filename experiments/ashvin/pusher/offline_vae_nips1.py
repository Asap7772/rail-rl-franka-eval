import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v0
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.grill.launcher import grill_her_td3_full_experiment

from railrl.launchers.arglauncher import run_variants

if __name__ == "__main__":
    variant = dict(
        imsize=84,
        init_camera=sawyer_pusher_camera_upright_v0,
        env_id='SawyerPushNIPS-v0',
        grill_variant=dict(
            save_video=True,
            save_video_period=100,
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
                    collection_mode='online-parallel',
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
                fraction_goals_are_rollout_goals=0.1,
                fraction_resampled_goals_are_env_goals=0.5,
            ),
            algorithm='RIG-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.5,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            )
        ),
        train_vae_variant=dict(
            vae_path=None,
            representation_size=16,
            beta=2.5,
            num_epochs=1000,
            dump_skew_debug_plots=False,
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=5000,
                oracle_dataset_using_set_to_goal=True,
                use_cached=True,
                vae_dataset_specific_kwargs=dict(
                ),
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
            save_period=100,
        ),
    )

    search_space = {
        'seedid': range(5),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer_pusher_offline_vae'

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(grill_her_td3_full_experiment, variants, run_id=0)
