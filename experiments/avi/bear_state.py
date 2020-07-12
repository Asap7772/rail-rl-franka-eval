import copy

import gym
import numpy as np
import torch.nn as nn

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.launchers.launcher_util import run_experiment
from railrl.samplers.data_collector import MdpPathCollector, CustomMdpPathCollector
from railrl.torch.networks import MlpQf, TanhMlpPolicy
from railrl.torch.sac.policies import (
    TanhGaussianPolicy, VAEPolicy
)
from railrl.torch.sac.bear import BEARTrainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import roboverse
import pickle


DEFAULT_BUFFER = ('/media/avi/data/Work/doodad_output/20-03-31-railrl-bullet-'
                  'SawyerReach-v0-state/20-03-31-railrl-bullet-SawyerReach-v0-state'
                  '_2020_03_31_15_37_01_id123824--s443957/buffers/epoch_450.pkl')


def experiment(variant):

    expl_env = roboverse.make(variant['env'], gui=False, randomize=True,
                              observation_mode='state', reward_type='shaped')
    eval_env = expl_env

    action_dim = int(np.prod(eval_env.action_space.shape))
    state_dim = obs_dim = 11
    M = 256

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

    vae_policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        latent_dim=action_dim * 2,
    )

    eval_path_collector = CustomMdpPathCollector(
        eval_env,
        save_images=True,
        **variant['eval_path_collector_kwargs']
    )

    vae_eval_path_collector = CustomMdpPathCollector(
        eval_env,
        max_num_epoch_paths_saved=5,
        save_images=True,
    )

    with open(variant['buffer'], 'rb') as f:
        replay_buffer = pickle.load(f)

    trainer = BEARTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vae=vae_policy,
        **variant['trainer_kwargs']
    )

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
        vae_evaluation_data_collector=vae_eval_path_collector,
        replay_buffer=replay_buffer,
        q_learning_alg=True,
        batch_rl=variant['batch_rl'],
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
            target_update_method='default',

            # use_automatic_entropy_tuning=True,
            # BEAR specific params
            mode='auto',
            kernel_choice='laplacian',
            policy_update_style='0',  # 0 is min, 1 is average (for double Qs)
            mmd_sigma=20.0,  # 5, 10, 40, 50

            target_mmd_thresh=0.05,  # .1, .07, 0.01, 0.02

            # gradient penalty hparams
            with_grad_penalty_v1=False,
            with_grad_penalty_v2=False,
            grad_coefficient_policy=0.001,
            grad_coefficient_q=1E-4,
            start_epoch_grad_penalty=24000,

            # Target nets/ policy vs Q-function update
            use_target_nets=True,
            policy_update_delay=1,
            num_steps_policy_update_only=1,

            ## advantage weighting
            use_adv_weighting=False,
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
        batch_rl=True,

    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='SawyerReach-v0',
                        choices=('SawyerReach-v0', 'SawyerGraspOne-v0'))
    parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    args = parser.parse_args()

    variant['env'] = args.env
    variant['obs'] = 'state'
    variant['buffer'] = args.buffer

    n_seeds = 1
    mode = 'local'
    # exp_prefix = 'dev-{}'.format(
    #     __file__.replace('/', '-').replace('_', '-').split('.')[0]
    # )
    exp_prefix = 'railrl-BEAR-bullet-{}-state'.format(args.env)

    # n_seeds = 5
    # mode = 'ec2'

    search_space = {
        'shared_qf_conv': [
            True,
            # False,
        ],
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
                gpu_id=0,
                unpack_variant=False,
            )
