from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm


class TD3EnsembleQs(TorchRLAlgorithm):
    """
    Twin Delayed Deep Deterministic policy gradients
    """

    def __init__(
            self,
            env,
            qf1,
            qf2,
            policy,
            exploration_policy,
            eval_policy=None,

            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,

            policy_learning_rate=1e-3,
            qf_learning_rate=1e-3,
            policy_and_target_update_period=2,
            tau=0.005,
            qf_criterion=None,
            optimizer_class=optim.Adam,

            ensemble_qs=None,

            **kwargs
    ):
        super().__init__(
            env,
            exploration_policy,
            eval_policy=eval_policy or policy,
            **kwargs
        )
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf1 = qf1
        self.qf2 = qf2
        self.policy = policy

        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip

        self.policy_and_target_update_period = policy_and_target_update_period
        self.tau = tau
        self.qf_criterion = qf_criterion

        self.target_policy = policy.copy()
        self.target_qf1 = self.qf1.copy()
        self.target_qf2 = self.qf2.copy()
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_learning_rate,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_learning_rate,
        )
        self.ensemble_qs = ensemble_qs if ensemble_qs is not None else []
        self.ensemble_q_optimizers = [optimizer_class(
            q.parameters(),
            lr=qf_learning_rate,
        ) for q in self.ensemble_qs]
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_learning_rate,
        )

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        self._train_given_data(
            rewards,
            terminals,
            obs,
            actions,
            next_obs,
        )

    def _train_given_data(
        self,
        rewards,
        terminals,
        obs,
        actions,
        next_obs,
        logger_prefix="",
    ):
        """
        Critic operations.
        """

        next_actions = self.target_policy(next_obs)
        noise = torch.normal(
            torch.zeros_like(next_actions),
            self.target_policy_noise,
        )
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

        for q, o in zip(self.ensemble_qs, self.ensemble_q_optimizers):
            q_pred = q(obs, actions)
            bellman_errors_q = (q_pred - q_target) ** 2
            q_loss = bellman_errors_q.mean()

            o.zero_grad()
            q_loss.backward()
            o.step()

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
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            policy_actions = self.policy(obs)
            q_output = self.qf1(obs, policy_actions)
            policy_loss = - q_output.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            if policy_loss is None:
                policy_actions = self.policy(obs)
                q_output = self.qf1(obs, policy_actions)
                policy_loss = - q_output.mean()

            self.eval_statistics[logger_prefix + 'QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics[logger_prefix + 'QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics[logger_prefix + 'Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Bellman Errors 1',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Bellman Errors 2',
                ptu.get_numpy(bellman_errors_2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        self.update_epoch_snapshot(snapshot)
        return snapshot

    def update_epoch_snapshot(self, snapshot):
        snapshot.update(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.eval_policy,
            trained_policy=self.policy,
            target_policy=self.target_policy,
            exploration_policy=self.exploration_policy,
            ensemble_qs=self.ensemble_qs,
        )

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_policy,
            self.target_qf1,
            self.target_qf2,
        ]  + self.ensemble_qs
