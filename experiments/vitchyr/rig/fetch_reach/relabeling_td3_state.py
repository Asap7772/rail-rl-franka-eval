import railrl.misc.hyperparameter as hyp
from railrl.launchers.experiments.vitchyr.multiworld import her_td3_experiment
from railrl.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=300,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=50,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=10000,
                reward_scale=100,
                render=False,
            ),
            her_kwargs=dict(),
            td3_kwargs=dict(),
        ),
        env_id='FetchReach-v1',
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='HER-TD3',
        version='normal',
        es_kwargs=dict(
            max_sigma=.2,
            epsilon=.3,
        ),
        exploration_type='gaussian_and_epsilon',
        observation_key='observation',
        desired_goal_key='desired_goal',
    )
    search_space = {
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [
            1,
            2,
            4,
            8,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 3
    mode = 'sss'
    exp_prefix = 'fetch-reach-her-td3'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                time_in_mins=23*60,
                snapshot_mode='gap_and_last',
                snapshot_gap=100,
            )
