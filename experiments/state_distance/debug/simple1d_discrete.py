import random

import numpy as np

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.simple1d import Simple1D, Simple1DTdmDiscretePlotter
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.dqn.policy import ArgmaxDiscretePolicy
from railrl.state_distance.old.discrete_action_networks import \
    VectorizedDiscreteQFunction, ArgmaxDiscreteTdmPolicy
from railrl.state_distance.tdm_ddpg import TdmDdpg
from railrl.torch.modules import HuberLoss
from railrl.torch.networks import FlattenMlp


def experiment(variant):
    env = variant['env_class']()

    obs_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    vectorized = variant['algo_params']['tdm_kwargs']['vectorized']
    if vectorized:
        qf = VectorizedDiscreteQFunction(
            observation_dim=int(np.prod(env.observation_space.low.shape)),
            action_dim=env.action_space.n,
            goal_dim=env.goal_dim,
            **variant['qf_params']
        )
        policy = ArgmaxDiscreteTdmPolicy(
            qf,
            **variant['policy_params']
        )
    else:
        qf = FlattenMlp(
            input_size=int(np.prod(env.observation_space.shape)) + env.goal_dim + 1,
            output_size=env.action_space.n,
            **variant['qf_params']
        )
        policy = ArgmaxDiscretePolicy(qf)
    es = OUStrategy(
        action_space=env.action_space,
        theta=0.1,
        max_sigma=0.1,
        min_sigma=0.1,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_params']
    )
    qf_criterion = variant['qf_criterion_class'](
        **variant['qf_criterion_params']
    )
    algo_params = variant['algo_params']
    algo_params['ddpg_kwargs']['qf_criterion'] = qf_criterion
    plotter = Simple1DTdmDiscretePlotter(
        tdm=qf,
        location_lst=np.array([-5, 0, 5]),
        goal_lst=np.array([-5, 0, 5]),
        max_tau=algo_params['tdm_kwargs']['max_tau'],
        grid_size=10,
    )
    algo_params['ddpg_kwargs']['plotter'] = plotter
    algorithm = TdmDdpg(
        env,
        qf=qf,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **algo_params
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "simple-1d-continuous"

    num_epochs = 100
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 30

    # noinspection PyTypeChecker
    max_tau = 5
    variant = dict(
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=num_epochs,
                num_steps_per_epoch=num_steps_per_epoch,
                num_steps_per_eval=num_steps_per_eval,
                max_path_length=max_path_length,
                num_updates_per_env_step=1,
                batch_size=64,
                discount=1,
                save_replay_buffer=True,
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=True,
                cycle_taus_for_rollout=True,
                max_tau=max_tau,
            ),
            ddpg_kwargs=dict(
                # tau=0.01,
                tau=1,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
            ),
        ),
        her_replay_buffer_params=dict(
            max_size=int(5E4),
            num_goals_to_sample=4,
        ),
        qf_params=dict(
            hidden_sizes=[100, 100],
            max_tau=max_tau,
        ),
        policy_params=dict(
            hidden_sizes=[100, 100],
        ),
        qf_criterion_class=HuberLoss,
        qf_criterion_params=dict(),
        version="DDPG-TDM",
        algorithm="DDPG-TDM",
    )
    search_space = {
        'env_class': [
            Simple1D,
        ],
        'algo_params.tdm_kwargs.vectorized': [
            False,
        ],
        'algo_params.tdm_kwargs.sample_rollout_goals_from': [
            'environment',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
                exp_prefix=exp_prefix,
                mode=mode,
            )
