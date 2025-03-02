from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
from railrl.envs.multitask.pusher2d import FullPusher2DEnv

from railrl.launchers.arglauncher import run_variants
import railrl.misc.hyperparameter as hyp
from railrl.torch.vae.vae_experiment import experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    vae_paths = {
        "2": "/home/ashvin/data/s3doodad/ashvin/vae/new-reacher2d-random/run0/id0/params.pkl",
        "4": "/home/ashvin/data/s3doodad/ashvin/vae/new-reacher2d-random/run0/id1/params.pkl",
        "8": "/home/ashvin/data/s3doodad/ashvin/vae/new-reacher2d-random/run0/id2/params.pkl",
        "16": "/home/ashvin/data/s3doodad/ashvin/vae/new-reacher2d-random/run0/id3/params.pkl"
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
            arm_range=2,
        ),
        algorithm='TD3',
        normalize=False,
        rdim=4,
        render=False,
        env=FullPusher2DEnv,
        use_env_goals=True,
        vae_paths=vae_paths,
        wrap_mujoco_env=True,
        track_qpos_goal=5,
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
    run_variants(experiment, sweeper.iterate_hyperparameters(), run_id=3)
