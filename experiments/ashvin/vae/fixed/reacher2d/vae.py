from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
from railrl.envs.multitask.pusher2d import FullPusher2DEnv

from railrl.launchers.arglauncher import run_variants
import railrl.misc.hyperparameter as hyp
from railrl.torch.vae.vae_experiment import experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    vae_paths = {
        "2": "ashvin/vae/new-reacher2d/run1/id0/params.pkl",
        "4": "ashvin/vae/new-reacher2d/run1/id1/params.pkl",
        "8": "ashvin/vae/new-reacher2d/run1/id2/params.pkl",
        "16": "ashvin/vae/new-reacher2d/run1/id3/params.pkl"
    }

    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            # qf_learning_rate=1e-3,
            # policy_learning_rate=1e-4,
        ),
        env_kwargs=dict(
            ignore_multitask_goal=True,
            include_puck=False,
            arm_range=1.0,
        ),
        algorithm='TD3',
        normalize=False,
        rdim=4,
        render=False,
        env=FullPusher2DEnv,
        use_env_goals=False,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        track_qpos_goal=5,
        do_state_based_exp=False,
    )

    n_seeds = 3

    search_space = {
        'exploration_type': [
            'gaussian'
        ],
        'algo_kwargs.reward_scale': [0.01, 0.1, 1, 10],
        'algo_kwargs.discount': [0.99],
        'rdim': [2, 4, 8, 16],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=2)
