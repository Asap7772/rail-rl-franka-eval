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
from railrl.torch.networks import MlpQf, TanhMlpPolicy
from railrl.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicy,
)
from railrl.torch.sac.sac import SACTrainer
from railrl.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

from roboverse.envs.pointmass import PointmassBaseEnv


def experiment(variant):

    expl_env = PointmassBaseEnv()
    eval_env = expl_env

    action_dim = int(np.prod(eval_env.action_space.shape))
    state_dim = 3

    qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
    qf_kwargs['output_size'] = 1
    qf_kwargs['input_size'] = action_dim + state_dim
    qf1 = MlpQf(**qf_kwargs)
    qf2 = MlpQf(**qf_kwargs)

    target_qf_kwargs = copy.deepcopy(qf_kwargs)
    target_qf1 = MlpQf(**target_qf_kwargs)
    target_qf2 = MlpQf(**target_qf_kwargs)

    policy_kwargs = copy.deepcopy(variant['policy_kwargs'])
    policy_kwargs['action_dim'] = action_dim
    policy_kwargs['obs_dim'] = state_dim
    policy = TanhGaussianPolicy(**policy_kwargs)

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
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        algo_kwargs=dict(
            batch_size=256,
            max_path_length=20,
            num_epochs=500,
            num_eval_steps_per_epoch=1000,
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

        # replay_buffer_size=int(1E6),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        replay_buffer_size=int(5E5),
        expl_path_collector_kwargs=dict(),
        eval_path_collector_kwargs=dict(),
        shared_qf_conv=False,
        use_robot_state=True,
    )
    #
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str,
    #                     choices=('SawyerReach-v0', 'SawyerGraspOne-v0'))
    # parser.add_argument("--obs", type=str, choices=('pixels', 'pixels_debug'))
    # args = parser.parse_args()

    # variant['env'] = args.env
    # variant['obs'] = 'state'

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )
    exp_prefix = 'railrl-bullet-pointmass-state'

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
                gpu_id=1,
                unpack_variant=False,
            )
