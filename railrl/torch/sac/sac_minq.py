from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd


class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            behavior_policy=None,
            dim_mult=1,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            use_target_nets=True,
            policy_eval_start=0,
            num_qs=2,

            ## For min_Q runs
            with_min_q=False,
            new_min_q=False,
            min_q_version=0,
            temp=1.0,
            hinge_bellman=False,
            use_projected_grad=False,
            normalize_magnitudes=False,
            regress_constant=False,
            min_q_weight=1.0,
            data_subtract=True,

            ## sort of backup
            max_q_backup=False,
            deterministic_backup=False,
            num_random=4,

            ## handle lagrange
            with_lagrange=False,
            lagrange_thresh=10.0,

            ## Handling discrete actions
            discrete=False,
            *args, **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                if self.env is None:
                    self.target_entropy = -2
                else:
                    self.target_entropy = -np.prod(
                        self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.with_lagrange = with_lagrange 
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime,],
                lr=qf_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self._use_target_nets = use_target_nets
        self.policy_eval_start = policy_eval_start

        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self.policy_eval_start = policy_eval_start
        self._num_policy_steps = 1

        if not self._use_target_nets:
            self.target_qf1 = qf1
            self.target_qf2 = qf2

        self.softmax = torch.nn.Softmax(dim=-1)
        self.num_qs = num_qs
        
        ## min Q
        self.with_min_q = with_min_q
        self.new_min_q = new_min_q
        self.temp = temp
        self.min_q_version = min_q_version
        self.use_projected_grad = use_projected_grad
        self.normalize_magnitudes = normalize_magnitudes
        self.regress_constant = regress_constant
        self.min_q_weight = min_q_weight
        self.softmax = torch.nn.Softmax(dim=1)
        self.hinge_bellman = hinge_bellman
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)
        self.data_subtract = data_subtract

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random

        self.discrete = discrete

    def compute_new_grad(self, grad1, grad2):
        new_grad = []
        for (grad1i, grad2i) in zip(grad1, grad2):
            proj_i = ((grad1i * grad2i).sum() * grad2i) / (grad2i * grad2i + 1e-7).sum()
            # conditional =
            if self.normalize_magnitudes:
                proj1 = (grad1i - proj_i).clamp_(max=0.01, min=-0.01)
                new_grad.append(proj1 + grad2i)
            else:
                new_grad.append(grad1i - proj_i + grad2i)
        return new_grad

    def compute_mt_grad(self, grad1, grad2):
        """Solution from Koltun paper."""
        new_grad = []
        for (grad1i, grad2i) in zip(grad1, grad2):
            l2_norm_grad = torch.norm(grad1i - grad2i).pow(2)
            alpha_i = ((grad2i - grad1i) * grad2i).sum() / (l2_norm_grad + 1e-7)
            alpha_i = alpha_i.clamp_(min=0.0, max=1.0)
            new_grad.append(grad1i * alpha_i + (1.0 - alpha_i) * grad2i)
        return new_grad

    def _get_tensor_values(self, img, actions, network=None):
        action_shape = actions.shape[0]
        
        obs_shape = img.shape[0]
        
        num_repeat = int(action_shape / obs_shape)
        img_temp = img.unsqueeze(1).repeat(1, num_repeat, 1,1,1).view(img.shape[0] * num_repeat, img.shape[1],img.shape[2],img.shape[3])
        
        obs_temp = None

        preds = network(img_temp, obs_temp, actions)
        preds = preds.view(obs_shape, num_repeat, 1)
        return preds

    def _get_policy_actions(self, img, num_actions, network=None):
        img_temp = img.unsqueeze(1).repeat(1, num_actions, 1,1,1).view(img.shape[0] * num_actions, img.shape[1],img.shape[2],img.shape[3])
        
        obs_temp = None
        
        new_obs_actions, _, _, new_obs_log_pi, *_ = network( 
            img_temp, obs_temp, reparameterize=True, return_log_prob=True,
        )

        obs_shape = img.shape[0]
        return new_obs_actions, new_obs_log_pi.view(obs_shape, num_actions, 1)

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        obs = obs.reshape((obs.shape[0],) + (3,48, 48))
        if not self.discrete:
            actions = batch['actions']
        else:
            actions = batch['actions'].argmax(dim=-1)
        next_obs = batch['next_observations']
        next_obs = next_obs.reshape((next_obs.shape[0],) + (3,48, 48))

        """
        Policy and Alpha Loss
        """
        if self.discrete:
            new_obs_actions, pi_probs, log_pi, entropies = self.policy(obs, None, return_log_prob=True)
            new_next_actions, pi_next_probs, new_log_pi, next_entropies = self.policy(next_obs, None, return_log_prob=True)
            q_vector = self.qf1.q_vector(obs)
            q2_vector = self.qf2.q_vector(obs)
            q_next_vector = self.qf1.q_vector(next_obs)
            q2_next_vector = self.qf2.q_vector(next_obs)
        else:
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs, None, reparameterize=True, return_log_prob=True,
            )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (
                        log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        if not self.discrete:
            if self.num_qs == 1:
                q_new_actions = self.qf1(obs, None, new_obs_actions)
            else:
                q_new_actions = torch.min(
                    self.qf1(obs, None, new_obs_actions),
                    self.qf2(obs, None, new_obs_actions),
                )

        if self.discrete:
            target_q_values = torch.min(q_vector, q2_vector)
            policy_loss = - ((target_q_values * pi_probs).sum(
                dim=-1) + alpha * entropies).mean()
        else:
            policy_loss = (alpha * log_pi - q_new_actions).mean()

        if self._current_epoch < self.policy_eval_start:
            """Start with BC"""
            policy_log_prob = self.policy.log_prob(obs, None, actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()
            # print ('Policy Loss: ', policy_loss.item())

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, None, actions)
        if self.num_qs > 1:
            q2_pred = self.qf2(obs, None, actions)

        # Make sure policy accounts for squashing functions like tanh correctly!
        if not self.discrete:
            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                next_obs, None, reparameterize=True, return_log_prob=True,
            )
            new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(
                obs, None, reparameterize=True, return_log_prob=True,
            )
        else:
            new_curr_actions, pi_curr_probs, new_curr_log_pi, new_curr_entropies = self.policy(
                obs, None, return_log_prob=True)

        if not self.max_q_backup:
            if not self.discrete:
                if self.num_qs == 1:
                    target_q_values = self.target_qf1(next_obs, None, new_next_actions)
                else:
                    target_q_values = torch.min(
                        self.target_qf1(next_obs, None, new_next_actions),
                        self.target_qf2(next_obs, None, new_next_actions),
                    )
            else:
                target_q_values = torch.min(
                    (self.target_qf1.q_vector(next_obs) * pi_next_probs).sum(
                        dim=-1),
                    (self.target_qf2.q_vector(next_obs) * pi_next_probs).sum(
                        dim=-1)
                )
                target_q_values = target_q_values.unsqueeze(-1)

            if not self.deterministic_backup:
                target_q_values = target_q_values - alpha * new_log_pi

        if self.max_q_backup:
            """when using max q backup"""
            if not self.discrete:
                next_actions_temp, _ = self._get_policy_actions(next_obs,
                                                                num_actions=10,
                                                                network=self.policy)
                target_qf1_values = \
                self._get_tensor_values(next_obs, next_actions_temp,
                                        network=self.target_qf1).max(1)[0].view(
                    -1, 1)
                target_qf2_values = \
                self._get_tensor_values(next_obs, next_actions_temp,
                                        network=self.target_qf2).max(1)[0].view(
                    -1, 1)
                target_q_values = torch.min(target_qf1_values,
                                            target_qf2_values)  # + torch.max(target_qf1_values, target_qf2_values) * 0.25
            else:
                target_qf1_values = \
                self.target_qf1.q_vector(next_obs).max(dim=-1)[0]
                target_qf2_values = \
                self.target_qf2.q_vector(next_obs).max(dim=-1)[0]
                target_q_values = torch.min(target_qf1_values,
                                            target_qf2_values).unsqueeze(-1)

        q_target = self.reward_scale * rewards + (
                    1. - terminals) * self.discount * target_q_values

        # Only detach if we are not using Bellman residual and not otherwise
        if self._use_target_nets:
            q_target = q_target.detach()

        qf1_loss = self.qf_criterion(q1_pred, q_target)
        if self.num_qs > 1:
            qf2_loss = self.qf_criterion(q2_pred, q_target)

        if self.hinge_bellman:
            qf1_loss = self.softplus(q_target - q1_pred).mean()
            qf2_loss = self.softplus(q_target - q2_pred).mean()

        ## add min_q
        if self.with_min_q:
            if not self.discrete:
                random_actions_tensor = torch.FloatTensor(
                    q2_pred.shape[0] * self.num_random,
                    actions.shape[-1]).uniform_(-1, 1).cuda()
                curr_actions_tensor, curr_log_pis = self._get_policy_actions(
                    obs, num_actions=self.num_random, network=self.policy)
                new_curr_actions_tensor, new_log_pis = self._get_policy_actions(
                    next_obs, num_actions=self.num_random, network=self.policy)
                q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
                q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
                q1_curr_actions = self._get_tensor_values(obs,curr_actions_tensor, network=self.qf1)
                q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
                q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf1)
                q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor,network=self.qf2)

                # q1_next_states_actions = self._get_tensor_values(next_obs, new_curr_actions_tensor, network=self.qf1)
                # q2_next_states_actions = self._get_tensor_values(next_obs, new_curr_actions_tensor, network=self.qf2)

                cat_q1 = torch.cat(
                    [q1_rand, q1_pred.unsqueeze(1), q1_next_actions,
                     q1_curr_actions], 1
                )
                cat_q2 = torch.cat(
                    [q2_rand, q2_pred.unsqueeze(1), q2_next_actions,
                     q2_curr_actions], 1
                )
                std_q1 = torch.std(cat_q1, dim=1)
                std_q2 = torch.std(cat_q2, dim=1)

                if self.min_q_version == 3:
                    # importance sammpled version
                    random_density = np.log(
                        0.5 ** curr_actions_tensor.shape[-1])
                    cat_q1 = torch.cat(
                        [q1_rand - random_density,
                         q1_next_actions - new_log_pis.detach(),
                         q1_curr_actions - curr_log_pis.detach()], 1
                    )
                    cat_q2 = torch.cat(
                        [q2_rand - random_density,
                         q2_next_actions - new_log_pis.detach(),
                         q2_curr_actions - curr_log_pis.detach()], 1
                    )

                if self.min_q_version == 0:
                    min_qf1_loss = cat_q1.mean() * self.min_q_weight
                    min_qf2_loss = cat_q2.mean() * self.min_q_weight
                elif self.min_q_version == 1:
                    """Expectation under softmax distribution"""
                    softmax_dist_1 = self.softmax(
                        cat_q1 / self.temp).detach() * self.temp
                    softmax_dist_2 = self.softmax(
                        cat_q2 / self.temp).detach() * self.temp
                    min_qf1_loss = (cat_q1 * softmax_dist_1).mean() * self.min_q_weight
                    min_qf2_loss = (cat_q2 * softmax_dist_2).mean() * self.min_q_weight
                elif self.min_q_version == 2 or self.min_q_version == 3:
                    """log sum exp for the min"""
                    min_qf1_loss = torch.logsumexp(cat_q1 / self.temp,
                                                   dim=1, ).mean() * self.min_q_weight * self.temp
                    min_qf2_loss = torch.logsumexp(cat_q2 / self.temp,
                                                   dim=1, ).mean() * self.min_q_weight * self.temp

                if self.data_subtract:
                    """Subtract the log likelihood of data"""
                    min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
                    min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight
            else:
                q1_policy = (q_vector * pi_probs).sum(dim=-1)
                q2_policy = (q2_vector * pi_probs).sum(dim=-1)
                q1_next_actions = (q_next_vector * pi_next_probs).sum(dim=-1)
                q2_next_actions = (q2_next_vector * pi_next_probs).sum(dim=-1)

                if self.min_q_version == 0:
                    min_qf1_loss = (q1_policy.mean() + q1_next_actions.mean() + q_vector.mean() + q_next_vector.mean()).mean() * self.min_q_weight
                    min_qf2_loss = (q2_policy.mean() + q1_next_actions.mean() + q2_vector.mean() + q2_next_vector.mean()).mean() * self.min_q_weight
                elif self.min_q_version == 1:
                    min_qf1_loss = (q_vector.mean() + q_next_vector.mean()).mean() * self.min_q_weight
                    min_qf2_loss = (q2_vector.mean() + q2_next_vector.mean()).mean() * self.min_q_weight
                else:
                    softmax_dist_q1 = self.softmax(
                        q_vector / self.temp).detach() * self.temp
                    softmax_dist_q2 = self.softmax(
                        q2_vector / self.temp).detach() * self.temp
                    min_qf1_loss = (q_vector * softmax_dist_q1).mean() * self.min_q_weight
                    min_qf2_loss = (q2_vector * softmax_dist_q2).mean() * self.min_q_weight

                if self.data_subtract:
                    min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
                    min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight

                std_q1 = torch.std(q_vector, dim=-1)
                std_q2 = torch.std(q2_vector, dim=-1)
                q1_on_policy = q1_policy.mean()
                q2_on_policy = q2_policy.mean()
                q1_random = q_vector.mean()
                q2_random = q2_vector.mean()
                q1_next_actions_mean = q1_next_actions.mean()
                q2_next_actions_mean = q2_next_actions.mean()
            
            if self.use_projected_grad:
                min_qf1_grad = torch.autograd.grad(min_qf1_loss,
                                                   inputs=[p for p in
                                                           self.qf1.parameters()],
                                                   create_graph=True,
                                                   retain_graph=True,
                                                   only_inputs=True
                                                   )
                min_qf2_grad = torch.autograd.grad(min_qf2_loss,
                                                   inputs=[p for p in
                                                           self.qf2.parameters()],
                                                   create_graph=True,
                                                   retain_graph=True,
                                                   only_inputs=True
                                                   )
                qf1_loss_grad = torch.autograd.grad(qf1_loss,
                                                    inputs=[p for p in
                                                            self.qf1.parameters()],
                                                    create_graph=True,
                                                    retain_graph=True,
                                                    only_inputs=True
                                                    )
                qf2_loss_grad = torch.autograd.grad(qf2_loss,
                                                    inputs=[p for p in
                                                            self.qf2.parameters()],
                                                    create_graph=True,
                                                    retain_graph=True,
                                                    only_inputs=True
                                                    )

                # this is for the offline setting
                # qf1_total_grad = self.compute_mt_grad(qf1_loss_grad, min_qf1_grad)
                # qf2_total_grad = self.compute_mt_grad(qf2_loss_grad, min_qf2_grad)
                qf1_total_grad = self.compute_new_grad(min_qf1_grad,
                                                       qf1_loss_grad)
                qf2_total_grad = self.compute_new_grad(min_qf2_grad,
                                                       qf2_loss_grad)
            else:
                if self.with_lagrange:	
                    alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0, max=2000000.0)	
                    orig_min_qf1_loss = min_qf1_loss	
                    orig_min_qf2_loss = min_qf2_loss	
                    min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)	
                    min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)	
                    self.alpha_prime_optimizer.zero_grad()	
                    alpha_prime_loss = -0.5 * (min_qf1_loss + min_qf2_loss)	
                    alpha_prime_loss.backward(retain_graph=True)	
                    self.alpha_prime_optimizer.step()
                qf1_loss = qf1_loss + min_qf1_loss
                qf2_loss = qf2_loss + min_qf2_loss

        """
        Update networks
        """
        # Update the Q-functions iff
        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        if self.with_min_q and self.use_projected_grad:
            for (p, proj_grad) in zip(self.qf1.parameters(), qf1_total_grad):
                p.grad.data = proj_grad
        self.qf1_optimizer.step()

        if self.num_qs > 1:
            self.qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph=True)
            if self.with_min_q and self.use_projected_grad:
                for (p, proj_grad) in zip(self.qf2.parameters(),
                                          qf2_total_grad):
                    p.grad.data = proj_grad
            self.qf2_optimizer.step()

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._use_target_nets:
            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    self.qf1, self.target_qf1, self.soft_target_tau
                )
                if self.num_qs > 1:
                    ptu.soft_update_from_to(
                        self.qf2, self.target_qf2, self.soft_target_tau
                    )
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            if not self.discrete:
                policy_loss = (log_pi - q_new_actions).mean()
            else:
                target_q_values = torch.min(q_vector, q2_vector)
                policy_loss = - ((target_q_values * pi_probs).sum(
                    dim=-1) + alpha * entropies).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            if self.num_qs > 1:
                self.eval_statistics['QF2 Loss'] = np.mean(
                    ptu.get_numpy(qf2_loss))

            if self.with_min_q and not self.discrete:
                self.eval_statistics['Std QF1 values'] = np.mean(
                    ptu.get_numpy(std_q1))
                self.eval_statistics['Std QF2 values'] = np.mean(
                    ptu.get_numpy(std_q2))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 in-distribution values',
                    ptu.get_numpy(q1_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 in-distribution values',
                    ptu.get_numpy(q2_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 random values',
                    ptu.get_numpy(q1_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 random values',
                    ptu.get_numpy(q2_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 next_actions values',
                    ptu.get_numpy(q1_next_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 next_actions values',
                    ptu.get_numpy(q2_next_actions),
                ))
            elif self.with_min_q and self.discrete:
                self.eval_statistics['Std QF1 values'] = np.mean(
                    ptu.get_numpy(std_q1))
                self.eval_statistics['Std QF2 values'] = np.mean(
                    ptu.get_numpy(std_q2))
                self.eval_statistics['QF1 on policy average'] = np.mean(
                    ptu.get_numpy(q1_on_policy))
                self.eval_statistics['QF2 on policy average'] = np.mean(
                    ptu.get_numpy(q2_on_policy))
                self.eval_statistics['QF1 random average'] = np.mean(
                    ptu.get_numpy(q1_random))
                self.eval_statistics['QF2 random average'] = np.mean(
                    ptu.get_numpy(q2_random))
                self.eval_statistics['QF1 next_actions_mean average'] = np.mean(
                    ptu.get_numpy(q1_next_actions_mean))
                self.eval_statistics['QF2 next_actions_mean average'] = np.mean(
                    ptu.get_numpy(q2_next_actions_mean))

            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics[
                'Num Policy Updates'] = self._num_policy_update_steps
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            if self.num_qs > 1:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q2 Predictions',
                    ptu.get_numpy(q2_pred),
                ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            if not self.discrete:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
            else:
                self.eval_statistics['Policy entropy'] = ptu.get_numpy(
                    entropies).mean()

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
            
            if self.with_lagrange:
                self.eval_statistics['Alpha Prime'] = alpha_prime.item()
                self.eval_statistics['Alpha Prime Loss'] = alpha_prime_loss.item()
                self.eval_statistics['Min Q1 Loss'] = orig_min_qf1_loss.item()
                self.eval_statistics['Min Q2 Loss'] = orig_min_qf2_loss.item()
            
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        return base_list

    def get_snapshot(self):
        return dict(trainer=self)
