from railrl.envs.wrappers import StackObservationEnv, RewardWrapperEnv
import railrl.torch.pytorch_util as ptu
from railrl.samplers.data_collector.step_collector import MdpStepCollector
from railrl.samplers.data_collector.path_collector import GoalConditionedPathCollector
from railrl.torch.networks import FlattenMlp
from railrl.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from railrl.torch.sac.awr_sac import AWRSACTrainer
from railrl.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

from railrl.demos.source.mdp_path_loader import MDPPathLoader
from railrl.visualization.video import save_paths

import torch

from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_and_epislon import GaussianAndEpislonStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from railrl.core import logger
from railrl.misc.asset_loader import load_local_or_remote_file

from railrl.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
from railrl.data_management.wrappers.concat_to_obs_wrapper import \
        ConcatToObsWrapper
from railrl.envs.reward_mask_wrapper import DiscreteDistribution, RewardMaskWrapper

def compute_hand_sparse_reward(next_obs, reward, done, info):
    return info['goal_achieved'] - 1

def resume(variant):
    data = load_local_or_remote_file(variant.get("pretrained_algorithm_path"), map_location="cuda")
    algo = data['algorithm']

    algo.num_epochs = variant['num_epochs']

    post_pretrain_hyperparams = variant["trainer_kwargs"].get("post_pretrain_hyperparams", {})
    algo.trainer.set_algorithm_weights(**post_pretrain_hyperparams)

    algo.train()

def process_args(variant):
    if variant.get("debug", False):
        variant['max_path_length'] = 50
        variant['batch_size'] = 5
        variant['num_epochs'] = 5
        variant['num_eval_steps_per_epoch'] = 100
        variant['num_expl_steps_per_train_loop'] = 100
        variant['num_trains_per_train_loop'] = 10
        variant['min_num_steps_before_training'] = 100

def experiment(variant):
    render = variant.get("render", False)
    debug = variant.get("debug", False)

    if variant.get("pretrained_algorithm_path", False):
        resume(variant)
        return

    env_class = variant["env_class"]
    env_kwargs = variant["env_kwargs"]
    expl_env = env_class(**env_kwargs)
    eval_env = env_class(**env_kwargs)
    env = eval_env

    if variant.get('sparse_reward', False):
        expl_env = RewardWrapperEnv(expl_env, compute_hand_sparse_reward)
        eval_env = RewardWrapperEnv(eval_env, compute_hand_sparse_reward)

    if variant.get('add_env_demos', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])

    if variant.get('add_env_offpolicy_data', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])

    if variant.get("use_masks", False):
        mask_wrapper_kwargs = variant.get("mask_wrapper_kwargs", dict())

        expl_mask_distribution_kwargs = variant["expl_mask_distribution_kwargs"]
        expl_mask_distribution = DiscreteDistribution(**expl_mask_distribution_kwargs)
        expl_env = RewardMaskWrapper(env, expl_mask_distribution, **mask_wrapper_kwargs)

        eval_mask_distribution_kwargs = variant["eval_mask_distribution_kwargs"]
        eval_mask_distribution = DiscreteDistribution(**eval_mask_distribution_kwargs)
        eval_env = RewardMaskWrapper(env, eval_mask_distribution, **mask_wrapper_kwargs)
        env = eval_env

    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    stack_obs = path_loader_kwargs.get("stack_obs", 1)
    if stack_obs > 1:
        expl_env = StackObservationEnv(expl_env, stack_obs=stack_obs)
        eval_env = StackObservationEnv(eval_env, stack_obs=stack_obs)

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = variant.get('achieved_goal_key', 'latent_achieved_goal')
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = eval_env.action_space.low.size

    if hasattr(expl_env, 'info_sizes'):
        env_info_sizes = expl_env.info_sizes
    else:
        env_info_sizes = dict()

    replay_buffer_kwargs=dict(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
    )
    replay_buffer_kwargs.update(variant.get('replay_buffer_kwargs', dict()))
    replay_buffer = ConcatToObsWrapper(
        ObsDictRelabelingBuffer(**replay_buffer_kwargs),
        ["resampled_goals", ],
    )
    replay_buffer_kwargs.update(variant.get('demo_replay_buffer_kwargs', dict()))
    demo_train_buffer = ConcatToObsWrapper(
        ObsDictRelabelingBuffer(**replay_buffer_kwargs),
        ["resampled_goals", ],
    )
    demo_test_buffer = ConcatToObsWrapper(
        ObsDictRelabelingBuffer(**replay_buffer_kwargs),
        ["resampled_goals", ],
    )

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

    expl_policy = policy
    exploration_kwargs =  variant.get('exploration_kwargs', {})
    if exploration_kwargs:
        if exploration_kwargs.get("deterministic_exploration", False):
            expl_policy = MakeDeterministic(policy)

        exploration_strategy = exploration_kwargs.get("strategy", None)
        if exploration_strategy is None:
            pass
        elif exploration_strategy == 'ou':
            es = OUStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        elif exploration_strategy == 'gauss_eps':
            es = GaussianAndEpislonStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],  # constant sigma
                epsilon=0,
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        else:
            error

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
        eval_path_collector = GoalConditionedPathCollector(
            eval_env,
            MakeDeterministic(policy),
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            render=render,
        )
        expl_path_collector = GoalConditionedPathCollector(
            expl_env,
            policy,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            render=render,
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

    algorithm.train()
