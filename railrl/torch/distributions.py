"""
Add custom distributions in addition to th existing ones
"""
import torch
from torch.distributions import Distribution, Normal, Categorical, OneHotCategorical
import railrl.torch.pytorch_util as ptu
import numpy as np

class GaussianMixture(Distribution):
    def __init__(self, normal_means, normal_stds, weights):
        self.num_gaussians = weights.shape[1]
        self.normal_means = normal_means
        self.normal_stds = normal_stds
        self.normal = Normal(normal_means, normal_stds)
        self.normals = [Normal(normal_means[:, :, i], normal_stds[:, :, i]) for i in range(self.num_gaussians)]
        self.weights = weights
        self.categorical = OneHotCategorical(self.weights[:, :, 0])

    def log_prob(self, value, ):
        log_p = [self.normals[i].log_prob(value) for i in range(self.num_gaussians)]
        log_p = torch.stack(log_p, -1)
        log_p = log_p.sum(dim=1)
        log_weights = torch.log(self.weights[:, :, 0])
        lp = log_weights + log_p
        m = lp.max(dim=1, keepdim=True)[0] # log-sum-exp numerical stability trick
        log_p_mixture = m + torch.log(torch.exp(lp - m).sum(dim=1, keepdim=True))
        return log_p_mixture

    def sample(self, ):
        z = self.normal.sample().detach()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def rsample(self, ):
        z = (
            self.normal_means +
            self.normal_stds *
                Normal(
                    ptu.zeros(self.normal_means.size()),
                    ptu.ones(self.normal_stds.size())
                ).sample()
        )
        z.requires_grad_()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def mean(self, ):
        """Misleading function name; this actually now samples the mean of the
        most likely component.
        c ~ argmax(C), returns mu_c

        This often computes the mode of the distribution, but not always.
        """
        c = ptu.zeros(self.weights.shape[:2])
        ind = torch.argmax(self.weights, dim=1) # [:, 0]
        c.scatter_(1, ind, 1)
        s = torch.matmul(self.normal_means, c[:, :, None])
        return torch.squeeze(s, 2)

        # same computation but iterative; seems much slower
        # r = ptu.zeros(self.normal_means.shape[:2])
        # ind = torch.argmax(self.weights, dim=1)[:, 0]
        # torch.scatter_
        # for i in range(len(r)):
        #     r[i, :] = self.normal_means[i, :, ind[i].item()]
        # return r

        # this computes the mean; not too meaningful for GMM
        # m = torch.matmul(self.normal_means, self.weights.detach()).detach()
        # return torch.squeeze(m, 2)

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73

        This formula is mathematically equivalent to log(1 - tanh(x)^2).

        Derivation:

        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            value = torch.clamp(value, -0.999999, 0.999999)
            # pre_tanh_value = torch.log(
                # (1+value) / (1-value)
            # ) / 2
            pre_tanh_value = torch.log(1+value) / 2 - torch.log(1-value) / 2
            # ) / 2
        return self.normal.log_prob(pre_tanh_value) - 2. * (
                ptu.from_numpy(np.log([2.]))
                - pre_tanh_value
                - torch.nn.functional.softplus(-2. * pre_tanh_value)
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
                Normal(
                    ptu.zeros(self.normal_mean.size()),
                    ptu.ones(self.normal_std.size())
                ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
