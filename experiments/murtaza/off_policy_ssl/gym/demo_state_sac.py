import gym

from railrl.demos.source.mdp_path_loader import MDPPathLoader
from railrl.data_management.env_replay_buffer import EnvReplayBuffer, AWREnvReplayBuffer
from railrl.envs.wrappers import NormalizedBoxEnv
import railrl.torch.pytorch_util as ptu
from railrl.samplers.data_collector import MdpPathCollector
from railrl.samplers.data_collector.step_collector import MdpStepCollector
from railrl.torch.networks import FlattenMlp
import railrl.misc.hyperparameter as hyp
from railrl.torch.sac.awr_sac import AWRSACTrainer
from railrl.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy, GaussianPolicy
from railrl.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)
from railrl.launchers.experiments.ashvin.awr_sac_rl import experiment

from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    Walker2dEnv,
    HopperEnv,
)

from railrl.launchers.launcher_util import run_experiment

ENV_PARAMS = {
    'half-cheetah': {  # 6 DoF
        'env_class': HalfCheetahEnv,
        'env_id':'HalfCheetah-v2',
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 1000,
        'demo_path':"demos/hc_action_noise_1000.npy",
        'bc_num_pretrain_steps':50000,
    },
    'hopper': {  # 6 DoF
        'env_class': HopperEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 1000,
        'demo_path':"demos/hopper_action_noise_1000.npy",
        'bc_num_pretrain_steps':500000,
        'env_id':'Hopper-v2',
    },
    'ant': {  # 6 DoF
        'env_class': AntEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
        'demo_path':"demos/ant_action_noise_1000.npy",
        'bc_num_pretrain_steps':500000,
        'env_id':'Ant-v2',
    },
    'walker': {  # 6 DoF
        'env_class': Walker2dEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
        'demo_path':"demos/walker_action_noise_1000.npy",
        'bc_num_pretrain_steps':100000,
        'env_id':'Walker2d-v2',
    },
}

def experiment(variant):
    env_params = ENV_PARAMS[variant['env']]
    variant.update(env_params)
    variant['path_loader_kwargs']['demo_path'] = env_params['demo_path']
    variant['trainer_kwargs']['bc_num_pretrain_steps'] = env_params['bc_num_pretrain_steps']

    if 'env_id' in env_params:
        expl_env = gym.make(env_params['env_id'])
        eval_env = gym.make(env_params['env_id'])
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    N = variant['num_layers']
    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M]*N,
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M] * N,
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M] * N,
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M] * N,
    )
    if variant.get('policy_class') == TanhGaussianPolicy:
        policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M] * N,
    )
    else:
        policy = GaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M] * N,
        max_log_std=0,
        min_log_std=-6,
        )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    replay_buffer = AWREnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        use_weights=variant['use_weights'],
        policy=policy,
        qf1=qf1,
        weight_update_period=variant['weight_update_period'],
        beta=variant['trainer_kwargs']['beta'],
    )
    trainer = AWRSACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    if variant['collection_mode'] == 'online':
        expl_path_collector = MdpStepCollector(
            expl_env,
            policy,
        )
        algorithm = TorchOnlineRLAlgorithm(
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
    else:
        expl_path_collector = MdpPathCollector(
            expl_env,
            policy,
        )
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
        variant['replay_buffer_size'],
        expl_env,
    )
    demo_test_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    path_loader_class = variant.get('path_loader_class', MDPPathLoader)
    path_loader = path_loader_class(trainer,
                                    replay_buffer=replay_buffer,
                                    demo_train_buffer=demo_train_buffer,
                                    demo_test_buffer=demo_test_buffer,
                                    **variant['path_loader_kwargs']
                                    )
    if variant.get('load_demos', False):
        path_loader.load_demos()
    if variant.get('pretrain_policy', False):
        trainer.pretrain_policy_with_bc()
    if variant.get('pretrain_rl', False):
        trainer.pretrain_q_with_bc_data()
    if variant.get('train_rl', True):
        algorithm.train()

if __name__ == "__main__":
    variant = dict(
        num_epochs=3000,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=512,
        replay_buffer_size=int(1E6),
        layer_size=256,
        num_layers=4,
        algorithm="SAC BC",
        version="normal",
        collection_mode='batch',
        sac_bc=True,
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            use_automatic_entropy_tuning=True,
            bc_num_pretrain_steps=1000000,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=10000,
            policy_weight_decay=1e-4,
            compute_bc=True,
            weight_loss=False,
            bc_weight=1.0,
            rl_weight=0.0,
            bc_loss_type='mse',
            pretraining_env_logging_period=100000,
        ),
        policy_kwargs=dict(
            hidden_sizes=[256]*4,
            max_log_std=0,
            min_log_std=-6,
        ),
        path_loader_kwargs=dict(
            demo_path=None
        ),
        weight_update_period=10000,
    )

    search_space = {
        'use_weights':[True],
        # 'weight_update_period':[1000, 10000],
        'trainer_kwargs.use_automatic_entropy_tuning':[False],
        # 'trainer_kwargs.bc_num_pretrain_steps':[1000],
        'trainer_kwargs.bc_weight':[1],
        'trainer_kwargs.alpha':[0],
        'trainer_kwargs.weight_loss':[True],
        'trainer_kwargs.beta':[
            10,
            # 100,
        ],
        'train_rl':[False],
        'pretrain_rl':[False],
        'load_demos':[True],
        'pretrain_policy':[True],
        'env': [
            # 'ant',
            'half-cheetah',
            # 'walker',
            # 'hopper',
        ],
        'policy_class':[
          # TanhGaussianPolicy,
          GaussianPolicy,
        ],
        'trainer_kwargs.bc_loss_type':[
            'mse',
        ],
        'trainer_kwargs.awr_loss_type':[
            'mse',
        ]

    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    # n_seeds = 2
    # mode = 'ec2'
    # exp_prefix = 'bc_hc_gym_v4'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                num_exps_per_instance=1,
                use_gpu=False,
                gcp_kwargs=dict(
                    preemptible=False,
                ),
            )
