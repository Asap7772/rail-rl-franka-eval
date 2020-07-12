from railrl.data_management.env_replay_buffer import EnvReplayBuffer
import railrl.torch.pytorch_util as ptu
import railrl.misc.hyperparameter as hyp

from railrl.samplers.data_collector import MdpPathCollector
from railrl.torch.sac.policies import GaussianPolicy, GaussianMixturePolicy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicy,
)
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from railrl.torch.networks import FlattenMlp
from railrl.torch.sac.awr_sac import AWRSACTrainer
from railrl.demos.source.mdp_path_loader import MDPPathLoader
from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from railrl.launchers.arglauncher import run_variants

import roboverse

import numpy as np
import os.path as osp
from railrl.core import logger

import torch

DEFAULT_BUFFER = ('/media/avi/data/Work/github/jannerm/bullet-manipulation/data'
                  '/feb25_SawyerReach-v0_state_2K_dense_reward_randomize_noise'
                  '_std_0.2/reach_demos_state_noise02_vector.npy')


def experiment(variant):
    expl_env = roboverse.make(variant['env'], gui=False, randomize=True,
                              observation_mode='state', reward_type='shaped',
                              transpose_image=True)
    eval_env = expl_env
    action_dim = int(np.prod(eval_env.action_space.shape))
    obs_dim = 11

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy_class = variant.get("policy_class", TanhGaussianPolicy)
    policy = policy_class(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs'],
    )

    buffer_policy = policy_class(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs'],
    )

    trainer = AWRSACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        buffer_policy=buffer_policy,
        **variant['trainer_kwargs']
    )

    expl_policy = policy
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )

    replay_buffer_kwargs = dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )

    replay_buffer = EnvReplayBuffer(**replay_buffer_kwargs)

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        num_epochs=variant['num_epochs'],
        num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
        num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
        num_trains_per_train_loop=variant['num_trains_per_train_loop'],
        min_num_steps_before_training=variant['min_num_steps_before_training'],
    )

    algorithm.to(ptu.device)


    demo_train_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )
    demo_test_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )

    path_loader_kwargs = variant.get("path_loader_kwargs", {})

    save_paths = None  # FIXME(avi)
    if variant.get('save_paths', False):
        algorithm.post_train_funcs.append(save_paths)
    if variant.get('load_demos', False):
        path_loader_class = variant.get('path_loader_class', MDPPathLoader)
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos()
    if variant.get('pretrain_policy', False):
        trainer.pretrain_policy_with_bc()
    if variant.get('pretrain_rl', False):
        trainer.pretrain_q_with_bc_data()
    if variant.get('save_pretrained_algorithm', False):
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        torch.save(data, open(pt_path, "wb"))
        torch.save(data, open(p_path, "wb"))
    if variant.get('train_rl', True):
        algorithm.train()


if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='SawyerReach-v0',
                        choices=('SawyerReach-v0', 'SawyerGraspOne-v0'))
    parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    parser.add_argument("--gpu", type=int, default=1)

    args = parser.parse_args()

    variant = dict(
        num_epochs=1001,
        num_eval_steps_per_epoch=250,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=50,
        batch_size=1024,
        replay_buffer_size=int(1E6),

        layer_size=256,
        policy_class=GaussianMixturePolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
            num_gaussians=1,
        ),

        algorithm="SAC",
        version="normal",
        collection_mode='batch',
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=0.1,
            use_automatic_entropy_tuning=False,
            alpha=0,
            compute_bc=False,

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=25000,
            policy_weight_decay=1e-4,
            q_weight_decay=0,
            bc_loss_type="mle",
            awr_loss_type="mle",

            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=1.0,

            post_bc_pretrain_hyperparams=dict(
                bc_weight=0.0,
                compute_bc=False,
            ),

            reward_transform_kwargs=None, # r' = r + 1
            terminal_transform_kwargs=None, # t = 0
        ),
        num_exps_per_instance=1,
        region='us-west-2',

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            obs_key="state_observation",
            demo_paths=[
                dict(
                    path=args.buffer,
                    obs_dict=False,
                    is_demo=True,
                ),
                # dict(
                #     path="demos/icml2020/hand/pen_bc5.npy",
                #     obs_dict=False,
                #     is_demo=False,
                #     train_split=0.9,
                # ),
            ],
        ),
        add_env_demos=True,
        add_env_offpolicy_data=True,

        # logger_variant=dict(
        #     tensorboard=True,
        # ),
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
        # save_pretrained_algorithm=True,
        # snapshot_mode="all",
    )

    variant['env'] = args.env

    search_space = {
        # 'env': ["pen-sparse-v0", "door-sparse-v0", "relocate-sparse-v0", ],
        'trainer_kwargs.bc_loss_type': ["mle"],
        'trainer_kwargs.awr_loss_type': ["mle"],
        'seedid': [1],
        'trainer_kwargs.beta': [0.1, 0.3, 0.5],
        'trainer_kwargs.reparam_weight': [0.0, ],
        'trainer_kwargs.awr_weight': [1.0],
        'trainer_kwargs.bc_weight': [1.0, ],
        'policy_kwargs.std_architecture': ["values", ],

        # 'trainer_kwargs.compute_bc': [True, ],
        'trainer_kwargs.awr_use_mle_for_vf': [False, ],
        'trainer_kwargs.awr_sample_actions': [False, ],
        'trainer_kwargs.awr_min_q': [True, ],

        # 'trainer_kwargs.q_weight_decay': [1e-5, 1e-4, ],

        'trainer_kwargs.reward_transform_kwargs': [None, ],
        'trainer_kwargs.terminal_transform_kwargs': [dict(m=0, b=0), ],
        # 'policy_kwargs.num_gaussians': [1, 2, ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    mode = 'local'
    exp_prefix = 'railrl-AWRSAC-{}-state'.format(args.env)
    n_seeds = 1


    from joblib import Parallel, delayed
    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)
    num_threads = len(variants)
    Parallel(n_jobs=num_threads)(delayed(run_experiment)(
        experiment,
        exp_name=exp_prefix,
        mode=mode,
        variant=variant,
        use_gpu=True,
        gpu_id=args.gpu,
        unpack_variant=False,
    ) for variant in variants)
