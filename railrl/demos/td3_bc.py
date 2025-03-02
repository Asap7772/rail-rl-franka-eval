"""Deprecated in favor of railrl/torch/sac/awr_sac.py"""

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.torch_rl_algorithm import TorchTrainer

from railrl.misc.asset_loader import load_local_or_remote_file

import random
from railrl.torch.core import np_to_pytorch_batch
from railrl.data_management.path_builder import PathBuilder

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from railrl.core import logger

import glob

class TD3BCTrainer(TorchTrainer):
    """
    Twin Delayed Deep Deterministic policy gradients
    """

    def __init__(
            self,
            env,
            qf1,
            qf2,
            policy,
            target_qf1,
            target_qf2,
            target_policy,
            demo_path,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            demo_off_policy_path=[],
            apply_her_to_demos=False,
            add_demo_latents=False,
            demo_train_split=0.9,
            add_demos_to_replay_buffer=True,
            bc_num_pretrain_steps=0,
            bc_batch_size=64,
            bc_weight=1.0,
            rl_weight=1.0,
            q_num_pretrain_steps=0,
            weight_decay=0,
            eval_policy=None,
            recompute_reward=False,

            reward_scale=1.0,
            discount=0.99,
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,

            policy_learning_rate=1e-3,
            qf_learning_rate=1e-3,
            target_update_period=2,
            policy_update_period=2,
            tau=0.005,
            qf_criterion=None,
            optimizer_class=optim.Adam,
            beta=1.0,

            awr_policy_update=False,

            demo_beta=1,
            max_steps_till_train_rl=0,
            env_info_key=None,
            obs_key=None,
            max_path_length=0,
            goal_conditioned=False,
            use_demo_awr=False,
            **kwargs
    ):
        super().__init__()
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf1 = qf1
        self.qf2 = qf2
        self.policy = policy
        self.env = env

        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip

        self.target_update_period = target_update_period
        self.policy_update_period = policy_update_period
        self.tau = tau
        self.qf_criterion = qf_criterion
        self.recompute_reward = recompute_reward

        self.target_policy = target_policy
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_learning_rate,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_learning_rate,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_learning_rate,
            weight_decay=weight_decay,
        )
        self.bc_batch_size = bc_batch_size
        self.bc_weight = bc_weight
        self.rl_weight = rl_weight

        self.discount = discount
        self.reward_scale = reward_scale

        self.add_demos_to_replay_buffer = add_demos_to_replay_buffer
        self.demo_train_split = demo_train_split
        self.replay_buffer = replay_buffer
        self.demo_train_buffer = demo_train_buffer
        self.demo_test_buffer = demo_test_buffer
        self.add_demo_latents = add_demo_latents
        self.apply_her_to_demos = apply_her_to_demos

        self.demo_path = demo_path
        self.demo_off_policy_path = demo_off_policy_path

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.beta = beta

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain_steps = q_num_pretrain_steps
        self.demo_trajectory_rewards = []
        self.update_policy = True
        self.awr_policy_update = awr_policy_update

        self.demo_beta = demo_beta
        self.use_demo_awr = use_demo_awr
        self.max_steps_till_train_rl = max_steps_till_train_rl
        self.env_info_key = env_info_key
        self.obs_key = obs_key
        self.max_path_length=max_path_length

        self.update_policy = True
        self.awr_policy_update = awr_policy_update
        self.goal_conditioned=goal_conditioned

    def _update_obs_with_latent(self, obs):
        obs = self.env.update_obs(obs)
        # latent_obs = self.env._encode_one(obs["image_observation"])
        # latent_goal = np.array([]) # self.env._encode_one(obs["image_desired_goal"])
        # obs['latent_observation'] = latent_obs
        # obs['latent_achieved_goal'] = latent_goal
        # obs['latent_desired_goal'] = latent_goal
        # obs['observation'] = latent_obs
        # obs['achieved_goal'] = latent_goal
        # obs['desired_goal'] = latent_goal
        return obs

    def load_path(self, path, replay_buffer):
        # print("Loading path: ", path)
        # print("Path len", len(path))
        # print("Path observations: ", type(path), type(path[0]), print(path[0].keys()))
        # import ipdb; ipdb.set_trace()
        # path = path[0]
        if self.apply_her_to_demos:
            final_achieved_goal = path["observations"][-1]["state_achieved_goal"].copy()
        rewards = []
        path_builder = PathBuilder()

        print("loading path, length", len(path["observations"]), len(path["actions"]))
        H = min(len(path["observations"]), len(path["actions"]))
        print("actions", np.min(path["actions"]), np.max(path["actions"]))

        # zs = []
        for i in range(H):
            ob = path["observations"][i]
            action = path["actions"][i]
            reward = path["rewards"][i]
            next_ob = path["next_observations"][i]
            terminal = path["terminals"][i]
            agent_info = path["agent_infos"][i]
            env_info = path["env_infos"][i]

            # zs.append(ob['latent_observation'])
            # goal = path["goal"]["state_desired_goal"][0, :]
            # import pdb; pdb.set_trace()
            # print(goal.shape, ob["state_observation"])
            # state_observation = np.concatenate((ob["state_observation"], goal))
            # action = action[:2]
            if self.add_demo_latents:
                self._update_obs_with_latent(ob)
                self._update_obs_with_latent(next_ob)
                reward = self.env.compute_reward(
                    action,
                    next_ob,
                )
            if self.apply_her_to_demos:
                ob["state_desired_goal"] = final_achieved_goal
                next_ob["state_desired_goal"] = final_achieved_goal
                reward = self.env.compute_reward(
                    action,
                    next_ob,
                )

            if self.recompute_reward:
                reward = self.env.compute_reward(
                    action,
                    next_ob,
                )

            reward = np.array([reward])
            rewards.append(reward)
            terminal = np.array([terminal]).reshape((1, ))
            path_builder.add_all(
                observations=ob,
                actions=action,
                rewards=reward,
                next_observations=next_ob,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )
        self.demo_trajectory_rewards.append(rewards)
        path = path_builder.get_all_stacked()
        replay_buffer.add_path(path)
        # self.env.initialize(zs)

    def load_demos(self, ):
        # Off policy
        if type(self.demo_off_policy_path) is list:
            for demo_pattern in self.demo_off_policy_path:
                for demo_path in glob.glob(demo_pattern):
                    print("loading off-policy path", demo_path)
                    self.load_demo_path(demo_path, False)
        else:
            if self.demo_off_policy_path is not None:
                self.load_demo_path(self.demo_off_policy_path, False)

        if type(self.demo_path) is list:
            for demo_path in self.demo_path:
                self.load_demo_path(demo_path)
        else:
            self.load_demo_path(self.demo_path)


    # Parameterize which demo is being tested (and all jitter variants)
    # If on_policy is False, we only add the demos to the
    # replay buffer, and not to the demo_test or demo_train buffers
    def load_demo_path(self, demo_path, on_policy=True):
        data = list(load_local_or_remote_file(demo_path))
        if not on_policy:
            data = [data]
        # random.shuffle(data)
        N = int(len(data) * self.demo_train_split)
        print("using", N, "paths for training")

        if self.add_demos_to_replay_buffer:
            for path in data[:N]:
                self.load_path(path, self.replay_buffer)

        if on_policy:
            for path in data[:N]:
                self.load_path(path, self.demo_train_buffer)
            for path in data[N:]:
                self.load_path(path, self.demo_test_buffer)

    def get_batch_from_buffer(self, replay_buffer):
        batch = replay_buffer.random_batch(self.bc_batch_size)
        batch = np_to_pytorch_batch(batch)
        # obs = batch['observations']
        # next_obs = batch['next_observations']
        # goals = batch['resampled_goals']
        # import ipdb; ipdb.set_trace()
        # batch['observations'] = torch.cat((
        #     obs,
        #     goals
        # ), dim=1)
        # batch['next_observations'] = torch.cat((
        #     next_obs,
        #     goals
        # ), dim=1)
        return batch

    def pretrain_policy_with_bc(self):
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'pretrain_policy.csv', relative_to_snapshot_dir=True
        )
        for i in range(self.bc_num_pretrain_steps):
            train_batch = self.get_batch_from_buffer(self.demo_train_buffer)
            train_o = train_batch["observations"]
            train_u = train_batch["actions"]
            if self.goal_conditioned:
                train_g = train_batch["resampled_goals"]
                train_o = torch.cat((train_o, train_g), dim=1)

            train_pred_u = self.policy(train_o)
            train_error = (train_pred_u - train_u) ** 2
            train_bc_loss = train_error.mean()

            policy_loss = self.bc_weight * train_bc_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            test_batch = self.get_batch_from_buffer(self.demo_test_buffer)
            test_o = test_batch["observations"]
            test_u = test_batch["actions"]

            if self.goal_conditioned:
                test_g = test_batch["resampled_goals"]
                test_o = torch.cat((test_o, test_g), dim=1)

            test_pred_u = self.policy(test_o)

            test_error = (test_pred_u - test_u) ** 2
            test_bc_loss = test_error.mean()

            train_loss_mean = np.mean(ptu.get_numpy(train_bc_loss))
            test_loss_mean = np.mean(ptu.get_numpy(test_bc_loss))

            stats = {
                "Train BC Loss": train_loss_mean,
                "Test BC Loss": test_loss_mean,
                "policy_loss": ptu.get_numpy(policy_loss),
                "batch": i,
            }
            logger.record_dict(stats)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)

            if i % 1000 == 0:
                logger.save_itr_params(i, {
                    "evaluation/policy": self.policy,
                    "evaluation/env": self.env.wrapped_env,
                })

        logger.remove_tabular_output(
            'pretrain_policy.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )

    def pretrain_q_with_bc_data(self):
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'pretrain_q.csv', relative_to_snapshot_dir=True
        )
        self.update_policy = False
        # first train only the Q function
        for i in range(self.q_num_pretrain_steps):
            self.eval_statistics = dict()
            self._need_to_update_eval_statistics = True

            train_data = self.replay_buffer.random_batch(128)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            if self.goal_conditioned:
                goals = train_data['resampled_goals']
                train_data['observations'] = torch.cat((obs, goals), dim=1)
                train_data['next_observations'] = torch.cat((next_obs, goals), dim=1)
            self.train_from_torch(train_data)

            logger.record_dict(self.eval_statistics)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)

        self.update_policy = True
        # then train policy and Q function together
        for i in range(self.q_num_pretrain_steps):
            self.eval_statistics = dict()
            self._need_to_update_eval_statistics = True

            train_data = self.replay_buffer.random_batch(128)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            if self.goal_conditioned:
                goals = train_data['resampled_goals']
                train_data['observations'] = torch.cat((obs, goals), dim=1)
                train_data['next_observations'] = torch.cat((next_obs, goals), dim=1)
            self.train_from_torch(train_data)

            logger.record_dict(self.eval_statistics)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)

        logger.remove_tabular_output(
            'pretrain_q.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )

    def train_from_torch(self, batch):
        logger.push_tabular_prefix("train_q/")
        self.eval_statistics = dict()
        self._need_to_update_eval_statistics = True

        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        next_actions = self.target_policy(next_obs)
        noise = ptu.randn(next_actions.shape) * self.target_policy_noise
        noise = torch.clamp(
            noise,
            -self.target_policy_noise_clip,
            self.target_policy_noise_clip
        )
        noisy_next_actions = next_actions + noise

        target_q1_values = self.target_qf1(next_obs, noisy_next_actions)
        target_q2_values = self.target_qf2(next_obs, noisy_next_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        q1_pred = self.qf1(obs, actions)
        bellman_errors_1 = (q1_pred - q_target) ** 2
        qf1_loss = bellman_errors_1.mean()

        q2_pred = self.qf2(obs, actions)
        bellman_errors_2 = (q2_pred - q_target) ** 2
        qf2_loss = bellman_errors_2.mean()

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        policy_actions = policy_loss = None
        if self._n_train_steps_total % self.policy_update_period == 0:
            policy_actions = self.policy(obs)
            q_output = self.qf1(obs, policy_actions)

            if self.demo_train_buffer._size >= self.bc_batch_size:
                if self.use_demo_awr:
                    train_batch = self.get_batch_from_buffer(self.demo_train_buffer)
                    train_o = train_batch["observations"]
                    train_u = train_batch["actions"]
                    if self.goal_conditioned:
                        train_g = train_batch["resampled_goals"]
                        train_o = torch.cat((train_o, train_g), dim=1)
                    train_pred_u = self.policy(train_o)
                    train_error = (train_pred_u - train_u) ** 2
                    train_bc_loss = train_error.mean()

                    policy_q_output_demo_state = self.qf1(train_o, train_pred_u)
                    demo_q_output = self.qf1(train_o, train_u)

                    advantage = demo_q_output - policy_q_output_demo_state
                    self.eval_statistics['Train BC Loss'] = np.mean(ptu.get_numpy(
                        train_bc_loss
                    ))

                    if self.awr_policy_update:
                        train_bc_loss = (train_error * torch.exp((advantage) * self.demo_beta))
                        self.eval_statistics['Advantage'] = np.mean(ptu.get_numpy(advantage))

                    if self._n_train_steps_total < self.max_steps_till_train_rl:
                        rl_weight = 0
                    else:
                        rl_weight = self.rl_weight

                    policy_loss = - rl_weight * q_output.mean() + self.bc_weight * train_bc_loss.mean()
                else:
                    train_batch = self.get_batch_from_buffer(self.demo_train_buffer)

                    train_o = train_batch["observations"]
                    # train_pred_u = self.policy(train_o)
                    if self.goal_conditioned:
                        train_g = train_batch["resampled_goals"]
                        train_o = torch.cat((train_o, train_g), dim=1)
                    train_pred_u = self.policy(train_o)
                    train_u = train_batch["actions"]
                    train_error = (train_pred_u - train_u) ** 2
                    train_bc_loss = train_error.mean()

                    # Advantage-weighted regression
                    policy_error = (policy_actions - actions) ** 2
                    policy_error = policy_error.mean(dim=1)
                    advantage = q1_pred - q_output
                    weights = F.softmax((advantage / self.beta)[:, 0])

                    if self.awr_policy_update:
                        policy_loss = self.rl_weight * (policy_error * weights.detach() * self.bc_batch_size).mean()
                    else:
                        policy_loss = - self.rl_weight * q_output.mean() + self.bc_weight * train_bc_loss.mean()

                    self.eval_statistics.update(create_stats_ordered_dict(
                        'Advantage Weights',
                        ptu.get_numpy(weights),
                    ))

                self.eval_statistics['BC Loss'] = np.mean(ptu.get_numpy(
                    train_bc_loss
                ))


            else: # Normal TD3 update
                policy_loss = - self.rl_weight * q_output.mean()

            if self.update_policy and not (self.rl_weight==0 and self.bc_weight==0):
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))

        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            if policy_loss is None:
                policy_actions = self.policy(obs)
                q_output = self.qf1(obs, policy_actions)
                policy_loss = - q_output.mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 1',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 2',
                ptu.get_numpy(bellman_errors_2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

            if self.demo_test_buffer._size >= self.bc_batch_size:
                train_batch = self.get_batch_from_buffer(self.demo_train_buffer)
                train_u = train_batch["actions"]
                train_o = train_batch["observations"]
                if self.goal_conditioned:
                    train_g = train_batch["resampled_goals"]
                    train_o = torch.cat((train_o, train_g), dim=1)
                train_pred_u = self.policy(train_o)
                train_error = (train_pred_u - train_u) ** 2
                train_bc_loss = train_error

                policy_q_output_demo_state = self.qf1(train_o, train_pred_u)
                demo_q_output = self.qf1(train_o, train_u)

                train_advantage = demo_q_output - policy_q_output_demo_state

                test_batch = self.get_batch_from_buffer(self.demo_test_buffer)
                test_o = test_batch["observations"]
                test_u = test_batch["actions"]
                if self.goal_conditioned:
                    test_g = test_batch["resampled_goals"]
                    test_o = torch.cat((test_o, test_g), dim=1)
                test_pred_u = self.policy(test_o)
                test_error = (test_pred_u - test_u) ** 2
                test_bc_loss = test_error

                policy_q_output_demo_state = self.qf1(test_o, test_pred_u)
                demo_q_output = self.qf1(test_o, test_u)
                test_advantage = demo_q_output - policy_q_output_demo_state

                self.eval_statistics.update(create_stats_ordered_dict(
                    'Train BC Loss',
                    ptu.get_numpy(train_bc_loss),
                ))

                self.eval_statistics.update(create_stats_ordered_dict(
                    'Train Demo Advantage',
                    ptu.get_numpy(train_advantage),
                ))

                self.eval_statistics.update(create_stats_ordered_dict(
                    'Test BC Loss',
                    ptu.get_numpy(test_bc_loss),
                ))

                self.eval_statistics.update(create_stats_ordered_dict(
                    'Test Demo Advantage',
                    ptu.get_numpy(test_advantage),
                ))

        self._n_train_steps_total += 1

        logger.pop_tabular_prefix()

    def get_diagnostics(self):
        stats = super().get_diagnostics()

        # train_batch = self.get_batch_from_buffer(self.demo_train_buffer)
        # train_g = train_batch["resampled_goals"]

        # test_batch = self.get_batch_from_buffer(self.demo_test_buffer)
        # test_g = test_batch["resampled_goals"]

        # key_logs = np.zeros(5)
        # for i, goal in enumerate(train_g[:5]):
        #     o = self.env.reset()
        #     path_length = 0
        #     self.env.set_goal({'state_desired_goal': ptu.get_numpy(goal)})
        #     while path_length < self.max_path_length:
        #         o = o[self.obs_key]
        #         new_obs = np.hstack((o, goal))
        #         a, agent_info = self.policy.get_action(new_obs)
        #         next_o, r, d, env_info = self.env.step(a)
        #         path_length += 1
        #         if d:
        #             break
        #         o = next_o
        #     key_logs[i] = env_info[self.env_info_key]

        # self.eval_statistics.update(create_stats_ordered_dict(
        #     'Train Goals {}'.format(self.env_info_key),
        #     key_logs,
        # ))

        # key_logs = np.zeros(5)
        # for i, goal in enumerate(test_g[:5]):
        #     o = self.env.reset()
        #     path_length = 0
        #     self.env.set_goal({'state_desired_goal': ptu.get_numpy(goal)})
        #     while path_length < self.max_path_length:
        #         o = o[self.obs_key]
        #         new_obs = np.hstack((o, goal))
        #         a, agent_info = self.policy.get_action(new_obs)
        #         next_o, r, d, env_info = self.env.step(a)
        #         path_length += 1
        #         if d:
        #             break
        #         o = next_o
        #     key_logs[i] = env_info[self.env_info_key]

        # self.eval_statistics.update(create_stats_ordered_dict(
        #     'Test Goals {}'.format(self.env_info_key),
        #     key_logs,
        # ))

        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_policy,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            trained_policy=self.policy,
            target_policy=self.target_policy,
        )
