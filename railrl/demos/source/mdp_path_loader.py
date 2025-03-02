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

class MDPPathLoader:
    """
    Path loader for that loads obs-dict demonstrations
    into a Trainer with EnvReplayBuffer
    """

    def __init__(
            self,
            trainer,
            demo_path,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            demo_off_policy_path=[],
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
            env_info_key=None,
            obs_key=None,
            **kwargs
    ):
        self.trainer = trainer

        self.add_demos_to_replay_buffer = add_demos_to_replay_buffer
        self.demo_train_split = demo_train_split
        self.replay_buffer = replay_buffer
        self.demo_train_buffer = demo_train_buffer
        self.demo_test_buffer = demo_test_buffer

        self.demo_path = demo_path
        self.demo_off_policy_path = demo_off_policy_path

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain_steps = q_num_pretrain_steps
        self.demo_trajectory_rewards = []

        self.env_info_key = env_info_key
        self.obs_key = obs_key
        self.recompute_reward = recompute_reward

        self.trainer.replay_buffer = self.replay_buffer
        self.trainer.demo_train_buffer = self.demo_train_buffer
        self.trainer.demo_test_buffer = self.demo_test_buffer

    def load_path(self, path, replay_buffer):
        rewards = []
        path_builder = PathBuilder()

        print("loading path, length", len(path["observations"]), len(path["actions"]))
        H = min(len(path["observations"]), len(path["actions"]))
        print("actions", np.min(path["actions"]), np.max(path["actions"]))

        for i in range(H):
            ob = path["observations"][i]
            action = path["actions"][i]
            reward = path["rewards"][i]
            next_ob = path["next_observations"][i]
            terminal = path["terminals"][i]
            agent_info = path["agent_infos"][i]
            env_info = path["env_infos"][i]

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
        # if not on_policy:
            # data = [data]
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

