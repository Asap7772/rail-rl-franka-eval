from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
from railrl.envs.multitask.pusher2d import FullPusher2DEnv

from railrl.launchers.arglauncher import run_variants
import railrl.misc.hyperparameter as hyp
from railrl.torch.vae.vae_experiment import experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    vae_paths = {
        "2": "/home/ashvin/data/s3doodad/ashvin/vae/point2d-conv-sweep2/run1/id0/params.pkl",
        "4": "/home/ashvin/data/s3doodad/ashvin/vae/point2d-conv-sweep2/run1/id1/params.pkl",
        "8": "/home/ashvin/data/s3doodad/ashvin/vae/point2d-conv-sweep2/run1/id2/params.pkl",
        "16": "/home/ashvin/data/s3doodad/ashvin/vae/point2d-conv-sweep2/run1/id3/params.pkl"
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
            render_onscreen=False,
            render_size=84,
            ignore_multitask_goal=True,
            ball_radius=1,
        ),
        algorithm='TD3',
        normalize=False,
        rdim=4,
        render=False,
        env=MultitaskImagePoint2DEnv,
        use_env_goals=False,
        vae_paths=vae_paths,
    )

    n_seeds = 3

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.reward_scale': [1e-6, 1e-4, 1e-2, 1],
        'rdim': [2, 4, 8, 16],
        'seedid': range(n_seeds),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=0)
