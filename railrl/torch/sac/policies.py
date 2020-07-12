import numpy as np
import torch
from torch import nn as nn

from railrl.policies.base import ExplorationPolicy, Policy
from railrl.torch.core import eval_np
from railrl.torch.distributions import TanhNormal, Normal, GaussianMixture
from railrl.torch.networks import Mlp, CNN
from railrl.torch.vae.vae_base import GaussianLatentVAE

import railrl.torch.pytorch_util as ptu
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicyAdapter(nn.Module, ExplorationPolicy):
    """
    Usage:

    ```
    obs_processor = ...
    policy = TanhGaussianPolicyAdapter(obs_processor)
    ```
    """

    def __init__(
            self,
            obs_processor,
            obs_processor_output_dim,
            action_dim,
            hidden_sizes,
    ):
        super().__init__()
        self.obs_processor = obs_processor
        self.obs_processor_output_dim = obs_processor_output_dim
        self.mean_and_log_std_net = Mlp(
            hidden_sizes=hidden_sizes,
            output_size=action_dim*2,
            input_size=obs_processor_output_dim,
        )
        self.action_dim = action_dim

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_entropy: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        :param return_log_prob_of_mean: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        """
        h = self.obs_processor(obs)
        h = self.mean_and_log_std_net(h)
        mean, log_std = torch.split(h, self.action_dim, dim=1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        tanh_normal = TanhNormal(mean, std)
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        if return_entropy:
            entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
            # I'm not sure how to compute the (differential) entropy for a
            # tanh(Gaussian)
            entropy = entropy.sum(dim=1, keepdim=True)
            raise NotImplementedError()
        if return_log_prob_of_mean:
            tanh_normal = TanhNormal(mean, std)
            mean_action_log_prob = tanh_normal.log_prob(
                torch.tanh(mean),
                pre_tanh_value=mean,
            )
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)
        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value, tanh_normal
        )

    def log_prob_aviral(self, obs, actions):

        def atanh(x):
            one_plus_x = (1 + x).clamp(min=1e-6)
            one_minus_x = (1 - x).clamp(min=1e-6)
            return 0.5 * torch.log(one_plus_x / one_minus_x)

        raw_actions = atanh(actions)
        h = self.obs_processor(obs)
        h = self.mean_and_log_std_net(h)
        mean, log_std = torch.split(h, self.action_dim, dim=1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob.sum(-1)


# noinspection PyMethodOverriding
class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_entropy: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        :param return_log_prob_of_mean: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(ptu.device)
            log_std = torch.log(std) # self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        tanh_normal = TanhNormal(mean, std)
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        if return_entropy:
            entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
            # I'm not sure how to compute the (differential) entropy for a
            # tanh(Gaussian)
            entropy = entropy.sum(dim=1, keepdim=True)
            raise NotImplementedError()
        if return_log_prob_of_mean:
            tanh_normal = TanhNormal(mean, std)
            mean_action_log_prob = tanh_normal.log_prob(
                torch.tanh(mean),
                pre_tanh_value=mean,
            )
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)
        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value, tanh_normal
        )

    def logprob(self, action, mean, std):
        # import ipdb; ipdb.set_trace()
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def log_prob_aviral(self, obs, actions):

        def atanh(x):
            one_plus_x = (1 + x).clamp(min=1e-6)
            one_minus_x = (1 - x).clamp(min=1e-6)
            return 0.5 * torch.log(one_plus_x / one_minus_x)

        raw_actions = atanh(actions)
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob.sum(-1)

class GaussianPolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(ptu.zeros(action_dim, requires_grad=True))
            else:
                error
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_entropy: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        :param return_log_prob_of_mean: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            # log_std = self.last_fc_log_std(h)
            # log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                error
            log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(ptu.device)
            log_std = torch.log(std) # self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        normal = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            if return_log_prob:
                if reparameterize is True:
                    action = normal.rsample()
                else:
                    action = normal.sample()
                log_prob = normal.log_prob(action)
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = normal.rsample()
                else:
                    action = normal.sample()

        if return_entropy:
            entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
            # I'm not sure how to compute the (differential) entropy for a
            # tanh(Gaussian)
            entropy = entropy.sum(dim=1, keepdim=True)
            raise NotImplementedError()
        if return_log_prob_of_mean:
            normal = Normal(mean, std)
            mean_action_log_prob = normal.log_prob(mean)
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value, normal,
        )

class GaussianMixturePolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            num_gaussians=1,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim * num_gaussians,
            init_w=init_w,
            # output_activation=torch.tanh,
            **kwargs
        )
        self.action_dim = action_dim
        self.num_gaussians = num_gaussians
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]

            if self.std_architecture == "shared":
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim * num_gaussians)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(ptu.zeros(action_dim * num_gaussians, requires_grad=True))
            else:
                error
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        self.last_fc_weights = nn.Linear(last_hidden_size, num_gaussians)
        self.last_fc_weights.weight.data.uniform_(-init_w, init_w)
        self.last_fc_weights.bias.data.uniform_(-init_w, init_w)

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_entropy: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        :param return_log_prob_of_mean: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            # log_std = self.last_fc_log_std(h)
            # log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            # log_std = torch.sigmoid(self.last_fc_log_std(h))
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                error
            log_std = self.min_log_std + log_std * (self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(self.std)
            log_std = self.log_std

        weights = F.softmax(self.last_fc_weights(h)).reshape((-1, self.num_gaussians, 1))
        mixture_means = mean.reshape((-1, self.action_dim, self.num_gaussians, ))
        mixture_stds = std.reshape((-1, self.action_dim, self.num_gaussians, ))
        dist = GaussianMixture(mixture_means, mixture_stds, weights)

        # normal = Normal(mean, std)
        # import ipdb; ipdb.set_trace()

        mean = dist.mean()

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        if deterministic:
            action = mean
        else:
            # normal = Normal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action = dist.rsample()
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                if reparameterize is True:
                    action = dist.rsample()
                else:
                    action = dist.sample()

        if return_entropy:
            entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
            # I'm not sure how to compute the (differential) entropy for a
            # tanh(Gaussian)
            entropy = entropy.sum(dim=1, keepdim=True)
            raise NotImplementedError()
        if return_log_prob_of_mean:
            normal = Normal(mean, std)
            mean_action_log_prob = normal.log_prob(mean)
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, None, dist,
        )


class GaussianMixtureObsProcessorPolicy(GaussianMixturePolicy):
    def __init__(self, obs_processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_processor = obs_processor

    def forward(self, obs, *args, **kwargs):
        h_obs = self.obs_processor(obs)
        return super().forward(h_obs, *args, **kwargs)


class TanhGaussianObsProcessorPolicy(TanhGaussianPolicy):
    def __init__(self, obs_processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_obs_dim = obs_processor.input_size
        self.pre_goal_dim = obs_processor.input_size
        self.obs_processor = obs_processor

    def forward(self, obs, *args, **kwargs):
        obs_and_goal = obs
        assert obs_and_goal.shape[1] == self.pre_obs_dim + self.pre_goal_dim
        obs = obs_and_goal[:, :self.pre_obs_dim]
        goal = obs_and_goal[:, self.pre_obs_dim:]

        h_obs = self.obs_processor(obs)
        h_goal = self.obs_processor(goal)

        flat_inputs = torch.cat((h_obs, h_goal), dim=1)
        return super().forward(flat_inputs, *args, **kwargs)


# noinspection PyMethodOverriding
class TanhCNNGaussianPolicy(CNN, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            init_w=init_w,
            **kwargs
        )
        obs_dim = self.input_width * self.input_height
        action_dim = self.output_size
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(self.hidden_sizes) > 0:
                last_hidden_size = self.hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            return_entropy=False,
            return_log_prob_of_mean=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        :param return_entropy: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        :param return_log_prob_of_mean: If True, return the true expected log
        prob. Will not need to be differentiated through, so this can be a
        number.
        """
        h = super().forward(obs, return_last_activations=True)

        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        if return_entropy:
            entropy = log_std + 0.5 + np.log(2 * np.pi) / 2
            # I'm not sure how to compute the (differential) entropy for a
            # tanh(Gaussian)
            entropy = entropy.sum(dim=1, keepdim=True)
            raise NotImplementedError()
        if return_log_prob_of_mean:
            tanh_normal = TanhNormal(mean, std)
            mean_action_log_prob = tanh_normal.log_prob(
                torch.tanh(mean),
                pre_tanh_value=mean,
            )
            mean_action_log_prob = mean_action_log_prob.sum(dim=1, keepdim=True)
        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


class VAEPolicy(Mlp, ExplorationPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            latent_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.latent_dim = latent_dim

        self.e1 = torch.nn.Linear(obs_dim + action_dim, 750)
        self.e2 = torch.nn.Linear(750, 750)

        self.mean = torch.nn.Linear(750, self.latent_dim)
        self.log_std = torch.nn.Linear(750, self.latent_dim)

        self.d1 = torch.nn.Linear(obs_dim + self.latent_dim, 750)
        self.d2 = torch.nn.Linear(750, 750)
        self.d3 = torch.nn.Linear(750, action_dim)

        self.max_action = 1.0
        self.latent_dim = latent_dim

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic,
                       execute_actions=True)[0]

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * ptu.from_numpy(
            np.random.normal(0, 1, size=(std.size())))

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = ptu.from_numpy(np.random.normal(0, 1, size=(
            state.size(0), self.latent_dim))).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a))

    def decode_multiple(self, state, z=None, num_decode=10):
        if z is None:
            z = ptu.from_numpy(np.random.normal(0, 1, size=(
            state.size(0), num_decode, self.latent_dim))).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat(
            [state.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z],
            2)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a)), self.d3(a)


class ConvVAEPolicy(GaussianLatentVAE, ExplorationPolicy):
    """Conv vae policy"""

    def __init__(self, representation_size, architecture, action_dim,
                 encoder_class=CNN,
                 input_channels=1, imsize=48, init_w=1e-3, min_variance=1e-3,
                 hidden_init=ptu.fanin_init):
        super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))

        self.latent_dim = representation_size #FIXME(avi) Temp hack
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize * self.imsize * self.input_channels

        # deconv_args is also params for a convnet, since this policy is over a convnet
        conv_args, deconv_args = architecture['conv_args'], \
                                              architecture['deconv_args']
        conv_output_size = deconv_args['deconv_input_width'] * \
                           deconv_args['deconv_input_height'] * \
                           deconv_args['deconv_input_channels']

        # This is just for the image state encoder
        self.encoder = encoder_class(
            **conv_args,
            output_size=conv_output_size,
            init_w=init_w,
            hidden_init=hidden_init,
        )

        # Now we encode the actions as well
        self.action_encoder1 = torch.nn.Linear(
            self.encoder.output_size + action_dim, 750)
        self.action_encoder2 = torch.nn.Linear(750, representation_size)
        self.action_std_encoder = torch.nn.Linear(750, representation_size)

        self.action_std_decoder = torch.nn.Linear(representation_size, 750)

        self.action_encoder1.weight.data.uniform_(-init_w, init_w)
        self.action_encoder2.weight.data.uniform_(-init_w, init_w)
        self.action_std_decoder.weight.data.uniform_(-init_w, init_w)
        self.action_encoder1.bias.data.uniform_(-init_w, init_w)
        self.action_encoder2.bias.data.uniform_(-init_w, init_w)
        self.action_std_encoder.bias.data.uniform_(-init_w, init_w)

        # conv net for the observation input in the VAE decoder
        self.decoder = encoder_class(
            **conv_args,
            output_size=conv_output_size,
            init_w=init_w,
            hidden_init=hidden_init,
        )

        # For finally decoding the action
        self.action_decoder1 = torch.nn.Linear(
            self.decoder.output_size + representation_size, 750)
        self.action_decoder2 = torch.nn.Linear(750, 750)
        self.action_decoder3 = torch.nn.Linear(750, action_dim)

        self.representation_size = representation_size
        self.action_dim = action_dim

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic,
                       execute_actions=True)[0]

    def encode(self, input_obs, action):
        h = F.relu(self.encoder(input_obs))
        h_cat_action = torch.cat([h, action], dim=-1)
        x = F.relu(self.action_encoder1(h_cat_action))
        mu = self.action_encoder2(x)

        log_std = self.action_std_encoder(x)
        if self.log_min_variance is None:
            log_std = log_std
        else:
            log_std = self.log_min_variance + log_std
        return (mu, log_std)

    def decode(self, state, z=None):
        if z is None:
            z = ptu.from_numpy(np.random.normal(0, 1, size=(
            state.size(0), self.latent_dim))).clamp(-0.5, 0.5)

        h = F.relu(self.decoder(state))
        a = F.relu(self.action_decoder1(torch.cat([h, z], 1)))
        a = F.relu(self.action_decoder2(a))
        return torch.tanh(self.action_decoder3(a))

    def forward(self, state, action):
        mean, log_std = self.encode(state, action)

        # Clamped for numerical stability
        log_std = log_std.clamp(-4, 15)
        std = torch.exp(log_std)

        z = mean + std * ptu.from_numpy(
            np.random.normal(0, 1, size=(std.size())))
        u = self.decode(state, z)
        return u, mean, std

    def decode_multiple(self, state, z=None, num_decode=10):
        if z is None:
            z = ptu.from_numpy(np.random.normal(0, 1, size=(
            state.size(0), num_decode, self.latent_dim))).clamp(-0.5, 0.5)

        h = F.relu(self.decoder(state))
        a = F.relu(self.action_decoder1(torch.cat(
            [h.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.action_decoder2(a))
        return torch.tanh(self.action_decoder3(a)), self.action_decoder3(a)

    def logprob(self, inputs, obs_distribution_params):
        return None


class MakeDeterministic(Policy, ):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, *args, deterministic=False, **kwargs):
        return self.stochastic_policy.get_action(
            *args, deterministic=True, **kwargs
        )

    def to(self, device):
        self.stochastic_policy.to(device)

    def load_state_dict(self, stochastic_state_dict):
        self.stochastic_policy.load_state_dict(stochastic_state_dict)

    def state_dict(self):
        return self.stochastic_policy.state_dict()
