import railrl.torch.pytorch_util as ptu
from railrl.samplers.data_collector import MdpPathCollector, \
    CustomMdpPathCollector
from railrl.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from railrl.torch.sac.sac_minq import SACTrainer
from railrl.torch.networks import FlattenMlp
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment

import argparse, os
import numpy as np
import pickle

import roboverse

DEFAULT_BUFFER = ('/media/avi/data/Work/doodad_output/20-03-31-railrl-bullet-'
                  'SawyerReach-v0-state/20-03-31-railrl-bullet-SawyerReach-v0-state'
                  '_2020_03_31_15_37_01_id123824--s443957/buffers/epoch_450.pkl')

V2_ENVS = ['SawyerGraspV2-v0', 'SawyerGraspOneV2-v0', 'SawyerGraspTenV2-v0']
V4_ENVS = ['SawyerGraspOneV4-v0']
V5_ENVS = ['Widow200GraspV5-v0']

def experiment(variant):
    expl_env = roboverse.make(variant['env'], gui=False, randomize=True,
                              observation_mode='state', reward_type='shaped')
    eval_env = expl_env

    action_dim = int(np.prod(eval_env.action_space.shape))
    obs_dim = eval_env.observation_space.shape[0]

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M],  # Making it easier to visualize
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMdpPathCollector(
        expl_env,
    )

    with open(variant['buffer'], 'rb') as f:
        replay_buffer = pickle.load(f)

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        behavior_policy=None,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=True,
        batch_rl=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,  # halfcheetah_101000.pkl',
        load_buffer=None,
        env_name='Hopper-v2',
        sparse_reward=False,
        algorithm_kwargs=dict(
            # num_epochs=100,
            # num_eval_steps_per_epoch=50,
            # num_trains_per_train_loop=100,
            # num_expl_steps_per_train_loop=100,
            # min_num_steps_before_training=100,
            # max_path_length=10,
            num_epochs=3000,
            num_eval_steps_per_epoch=250,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=50,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            use_target_nets=True,
            policy_eval_start=40000,  # 30000,  (This is for offline)
            num_qs=2,

            # min Q
            with_min_q=True,
            new_min_q=True,
            hinge_bellman=False,
            temp=10.0,
            min_q_version=0,
            use_projected_grad=False,
            normalize_magnitudes=False,
            regress_constant=False,
            min_q_weight=1.0,
            data_subtract=True,

            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_buffer", action='store_true')
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_q_backup", type=str, default="False")
    parser.add_argument("--deterministic_backup", type=str, default="False")
    parser.add_argument("--policy_eval_start", default=40000, type=int)
    parser.add_argument('--use_projected_grad', default='False', type=str)
    parser.add_argument('--min_q_weight', default=1.0, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)
    parser.add_argument('--min_q_version', default=0, type=int)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument("--num-random", default=10, type=int)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    variant['trainer_kwargs']['use_projected_grad'] = (
        True if args.use_projected_grad == 'True' else False)
    variant['trainer_kwargs']['max_q_backup'] = (
        True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (
        True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['temp'] = args.temp
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['num_random'] = args.num_random

    variant['seed'] = args.seed

    variant['load_buffer'] = True
    variant['env'] = args.env
    variant['buffer'] = args.buffer

    if variant['env'] in V2_ENVS or variant['env'] in V5_ENVS:
        variant['algorithm_kwargs']['max_path_length'] = 20
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 100
    elif variant['env'] in V4_ENVS:
        variant['algorithm_kwargs']['max_path_length'] = 30
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 120

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'railrl-minQ-SAC-bullet-{}-state'.format(args.env)
    search_space = dict(shared_qf_conv=[True])

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
