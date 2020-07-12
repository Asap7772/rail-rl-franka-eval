import copy

import gym
import numpy as np
import torch.nn as nn

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.launchers.launcher_util import run_experiment
from railrl.pythonplusplus import identity
from railrl.samplers.data_collector import MdpPathCollector
from railrl.samplers.data_collector.step_collector import MdpStepCollector
from railrl.visualization.video import VideoSaveFunctionBullet

from railrl.torch.networks import (
    CNN,
    MlpQfWithObsProcessor,
    Split,
    FlattenEach,
    Concat,
    Flatten,
)
from railrl.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicyAdapter,
)
from railrl.torch.sac.sac import SACTrainer
from railrl.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

import roboverse
from multiworld.core.flat_goal_env import FlatEnv


def experiment(variant):

    expl_env = roboverse.make(variant['env'], gui=False, randomize=variant['randomize_env'],
                              observation_mode=variant['obs'], reward_type='shaped',
                              transpose_image=True)

    if variant['obs'] == 'pixels_debug':
        robot_state_dims = 11
    elif variant['obs'] == 'pixels':
        robot_state_dims = 4
    else:
        raise NotImplementedError

    expl_env = FlatEnv(expl_env, use_robot_state=variant['use_robot_state'],
                       robot_state_dims=robot_state_dims)
    eval_env = expl_env

    img_width, img_height = eval_env.image_shape
    num_channels = 3

    action_dim = int(np.prod(eval_env.action_space.shape))
    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=img_width,
        input_height=img_height,
        input_channels=num_channels,
    )
    if variant['use_robot_state']:
        cnn_params.update(
            added_fc_input_size=robot_state_dims,
            output_conv_channels=False,
            hidden_sizes=[400, 400],
            output_size=200,
        )
    else:
        cnn_params.update(
            added_fc_input_size=0,
            output_conv_channels=True,
            output_size=None,
        )
    qf_cnn = CNN(**cnn_params)

    if variant['use_robot_state']:
        qf_obs_processor = qf_cnn
    else:
        qf_obs_processor = nn.Sequential(
            qf_cnn,
            Flatten(),
        )

    qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
    qf_kwargs['obs_processor'] = qf_obs_processor
    qf_kwargs['output_size'] = 1

    if variant['use_robot_state']:
        qf_kwargs['input_size'] = (
                action_dim + qf_cnn.output_size
        )
    else:
        qf_kwargs['input_size'] = (
                action_dim + qf_cnn.conv_output_flat_size
        )

    qf1 = MlpQfWithObsProcessor(**qf_kwargs)
    qf2 = MlpQfWithObsProcessor(**qf_kwargs)

    target_qf_cnn = CNN(**cnn_params)
    if variant['use_robot_state']:
        target_qf_obs_processor = target_qf_cnn
    else:
        target_qf_obs_processor = nn.Sequential(
            target_qf_cnn,
            Flatten(),
        )

    target_qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
    target_qf_kwargs['obs_processor'] = target_qf_obs_processor
    target_qf_kwargs['output_size'] = 1

    if variant['use_robot_state']:
        target_qf_kwargs['input_size'] = (
                action_dim + target_qf_cnn.output_size
        )
    else:
        target_qf_kwargs['input_size'] = (
                action_dim + target_qf_cnn.conv_output_flat_size
        )

    target_qf1 = MlpQfWithObsProcessor(**target_qf_kwargs)
    target_qf2 = MlpQfWithObsProcessor(**target_qf_kwargs)

    action_dim = int(np.prod(eval_env.action_space.shape))
    policy_cnn = CNN(**cnn_params)
    if variant['use_robot_state']:
        policy_obs_processor = policy_cnn
    else:
        policy_obs_processor = nn.Sequential(
            policy_cnn,
            Flatten(),
        )

    if variant['use_robot_state']:
        policy = TanhGaussianPolicyAdapter(
            policy_obs_processor,
            policy_cnn.output_size,
            action_dim,
            **variant['policy_kwargs']
        )
    else:
        policy = TanhGaussianPolicyAdapter(
            policy_obs_processor,
            policy_cnn.conv_output_flat_size,
            action_dim,
            **variant['policy_kwargs']
        )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        **variant['eval_path_collector_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    if variant['collection_mode'] == 'batch':
        expl_path_collector = MdpPathCollector(
            expl_env,
            policy,
            **variant['expl_path_collector_kwargs']
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **variant['algo_kwargs']
        )
    elif variant['collection_mode'] == 'online':
        expl_path_collector = MdpStepCollector(
            expl_env,
            policy,
            **variant['expl_path_collector_kwargs']
        )
        algorithm = TorchOnlineRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **variant['algo_kwargs']
        )
    else:
        raise NotImplementedError

    video_func = VideoSaveFunctionBullet(variant)
    algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        trainer_kwargs=dict(
            discount=0.99,
            # soft_target_tau=5e-3,
            # target_update_period=1,
            soft_target_tau=1.0,
            target_update_period=1000,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        algo_kwargs=dict(
            batch_size=256,
            max_path_length=50,
            num_epochs=500,
            num_eval_steps_per_epoch=250,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=10*1000,
            # max_path_length=10,
            # num_epochs=100,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3],
            n_channels=[4, 4],
            strides=[1, 1],
            hidden_sizes=[32, 32],
            paddings=[1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2],
            pool_strides=[2, 2],
            pool_paddings=[0, 0],
        ),
        # replay_buffer_size=int(1E6),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
        replay_buffer_size=int(5E5),
        expl_path_collector_kwargs=dict(),
        eval_path_collector_kwargs=dict(),
        shared_qf_conv=False,
        use_robot_state=False,
        randomize_env=True,
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str,
                        choices=('SawyerReach-v0', 'SawyerGraspOne-v0'))
    parser.add_argument("--obs", type=str, choices=('pixels', 'pixels_debug'))
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    variant['env'] = args.env
    variant['obs'] = args.obs

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )
    exp_prefix = 'railrl-bullet-{}-{}'.format(args.env, args.obs)

    # n_seeds = 5
    # mode = 'ec2'
    # exp_prefix = 'railrl-bullet-sawyer-image-reach'

    search_space = {
        'shared_qf_conv': [
            True,
            # False,
        ],
        'collection_mode': [
            # 'batch',
            'online',
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_name=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                gpu_id=args.gpu,
                unpack_variant=False,
            )
