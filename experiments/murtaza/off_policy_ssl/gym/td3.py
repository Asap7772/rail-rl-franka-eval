"""
Test twin sac against various environments.
"""
from gym.envs.mujoco import (
    HalfCheetahEnv,
    AntEnv,
    Walker2dEnv,
    InvertedDoublePendulumEnv,
    HopperEnv,
    HumanoidEnv,
    SwimmerEnv,
)
from gym.envs.classic_control import PendulumEnv

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_and_epislon import GaussianAndEpislonStrategy
from railrl.launchers.launcher_util import run_experiment
import railrl.torch.pytorch_util as ptu
from railrl.samplers.data_collector import MdpPathCollector
from railrl.samplers.data_collector.step_collector import MdpStepCollector
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from railrl.torch.sac.sac import SACTrainer
import railrl.misc.hyperparameter as hyp
from railrl.torch.td3.td3 import TD3
from railrl.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

ENV_PARAMS = {
    'half-cheetah': {  # 6 DoF
        'env_class': HalfCheetahEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 1000,
    },
    'hopper': {  # 6 DoF
        'env_class': HopperEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 1000,
    },
    'humanoid': {  # 6 DoF
        'env_class': HumanoidEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
    },
    'inv-double-pendulum': {  # 2 DoF
        'env_class': InvertedDoublePendulumEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 100,
    },
    'pendulum': {  # 2 DoF
        'env_class': PendulumEnv,
        'num_expl_steps_per_train_loop': 200,
        'max_path_length': 200,
        'num_epochs': 200,
        'min_num_steps_before_training': 2000,
        'target_update_period': 200,
    },
    'ant': {  # 6 DoF
        'env_class': AntEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
    },
    'walker': {  # 6 DoF
        'env_class': Walker2dEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 3000,
    },
    'swimmer': {  # 6 DoF
        'env_class': SwimmerEnv,
        'num_expl_steps_per_train_loop': 1000,
        'max_path_length': 1000,
        'num_epochs': 2000,
    },
}


def experiment(variant):
    env_params = ENV_PARAMS[variant['env']]
    variant.update(env_params)

    expl_env = NormalizedBoxEnv(variant['env_class']())
    eval_env = NormalizedBoxEnv(variant['env_class']())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    es = GaussianAndEpislonStrategy(
        action_space=expl_env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )
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
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = TD3(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    if variant['collection_mode'] == 'online':
        expl_path_collector = MdpStepCollector(
            expl_env,
            expl_policy,
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
            expl_policy,
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
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        num_epochs=3000,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=256,
        replay_buffer_size=int(1E6),
        layer_size=256,
        algorithm="TD3",
        version="normal",
        collection_mode='batch',
        trainer_kwargs=dict(
            discount=0.99,
        ),
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # n_seeds = 5
    # mode = 'sss'
    # exp_prefix = 'railrl-half-cheetah-online'

    search_space = {
        'env': [
            'half-cheetah',
            # 'inv-double-pendulum',
            # 'pendulum',
            # 'ant',
            # 'walker',
            # 'hopper',
            # 'humanoid',
            # 'swimmer',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                time_in_mins=int(2.8 * 24 * 60),  # if you use mode=sss
            )