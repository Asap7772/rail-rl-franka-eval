import railrl.misc.hyperparameter as hyp
from railrl.envs.mujoco.sawyer_push_env import SawyerPushXYEnv
from railrl.launchers.experiments.vitchyr.multitask import ddpg_experiment
from railrl.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=300,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            render_during_eval=True,
        ),
        env_class=SawyerPushXYEnv,
        env_kwargs=dict(
            reward_info=dict(
                type='shaped',
            ),
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='DDPG',
        version='normal',
        normalize=True,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-sawyer-new-pusher'

    # n_seeds = 1
    # mode = 'ec2'
    # exp_prefix = 'sawyer-sim-push-easy-ish-check-fixed-env'

    search_space = {
        'algo_kwargs.max_path_length': [
            100,
        ],
        'exploration_type': [
            'ou',
            # 'epsilon',
            # 'gaussian',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                ddpg_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
            )
