"""Example behavior cloning script for pointmass.
If you are trying to run this code, ask Ashvin for the demonstration file:
demos/pointmass_demos_100.npy (which should go in your S3 storage)
"""

import railrl.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

from multiworld.envs.pygame.point2d import Point2DWallEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.launchers.arglauncher import run_variants

import numpy as np

def her_td3_experiment(variant):
    import gym
    import multiworld.envs.mujoco
    import multiworld.envs.pygame
    import railrl.samplers.rollout_functions as rf
    import railrl.torch.pytorch_util as ptu
    from railrl.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
    from railrl.exploration_strategies.ou_strategy import OUStrategy
    from railrl.torch.grill.launcher import get_video_save_func
    from railrl.demos.her_bc import HerBC
    from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
    from railrl.data_management.obs_dict_replay_buffer import (
        ObsDictRelabelingBuffer
    )

    if 'env_id' in variant:
        env = gym.make(variant['env_id'])
    else:
        env = variant['env_class'](**variant['env_kwargs'])

    observation_key = variant['observation_key']
    desired_goal_key = variant['desired_goal_key']
    variant['algo_kwargs']['her_kwargs']['observation_key'] = observation_key
    variant['algo_kwargs']['her_kwargs']['desired_goal_key'] = desired_goal_key
    if variant.get('normalize', False):
        raise NotImplementedError()

    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            **variant['es_kwargs']
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = HerBC(
        env,
        exploration_policy,
        policy,
        variant["demo_path"],
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    if variant.get("save_video", False):
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            policy,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    size = 0.1
    low = (-size, 0.4 - size, 0)
    high = (size, 0.4 + size, 0.1)
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=1001,
                num_steps_per_epoch=1,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=0,
                reward_scale=100,
                render=False,
                collection_mode='online',
                parallel_env_params=dict(
                    num_workers=1,
                ),
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            td3_kwargs=dict(
                weight_decay=0.0,
            ),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=1.0,
            fraction_resampled_goals_are_env_goals=0.0,
            ob_keys_to_save=['state_observation', 'state_desired_goal'],
        ),
        qf_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        policy_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        algorithm='HER-TD3',
        version='normal',
        es_kwargs=dict(
            max_sigma=.8,
        ),
        exploration_type='ou',
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        init_camera=sawyer_pusher_camera_upright_v2,
        do_state_exp=True,

        save_video=False,
        imsize=84,

        snapshot_mode='gap_and_last',
        snapshot_gap=50,

        env_class=Point2DWallEnv,
        env_kwargs=dict(
            render_onscreen=False,
            images_are_rgb=True,
        ),
        demo_path="demos/pointmass_demos_100.npy",

        num_exps_per_instance=1,
    )

    search_space = {
        # 'env_id': ['SawyerPushAndReacherXYEnv-v0', ],
        'seedid': range(3),
        'algo_kwargs.base_kwargs.num_updates_per_env_step': [1, ],
        'replay_buffer_kwargs.fraction_goals_are_rollout_goals': [0.0, 0.5, 1.0],
        'algo_kwargs.td3_kwargs.weight_decay': [0.0, 1e-3, 1e-4, 1e-5],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer_pusher_state_final'

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(her_td3_experiment, variants, run_id=6)
    # for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
    #     for i in range(n_seeds):
    #         run_experiment(
    #             her_td3_experiment,
    #             exp_prefix=exp_prefix,
    #             mode=mode,
    #             snapshot_mode='gap_and_last',
    #             snapshot_gap=50,
    #             variant=variant,
    #             use_gpu=True,
    #             num_exps_per_instance=5,
    #         )
