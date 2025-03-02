"""
Contain some self-contained modules. Maybe depend on pytorch_util.
"""
import torch
import torch.nn as nn
from railrl.torch import pytorch_util as ptu


class OuterProductLinear(nn.Module):
    def __init__(self, in_features1, in_features2, out_features, bias=True):
        super().__init__()
        self.fc = nn.Linear(
            (in_features1 + 1) * (in_features2 + 1),
            out_features,
            bias=bias,
        )

    def forward(self, in1, in2):
        out_product_flat = ptu.double_moments(in1, in2)
        return self.fc(out_product_flat)


class SelfOuterProductLinear(OuterProductLinear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, in_features, out_features, bias=bias)

    def forward(self, input):
        return super().forward(input, input)


class BatchSquareDiagonal(nn.Module):
    """
    Compute x^T diag(`diag_values`) x
    """
    def __init__(self, vector_size):
        super().__init__()
        self.vector_size = vector_size
        self.diag_mask = ptu.Variable(torch.diag(torch.ones(vector_size)),
                                      requires_grad=False)

    def forward(self, vector, diag_values):
        M = ptu.batch_diag(diag_values=diag_values, diag_mask=self.diag_mask)
        return ptu.batch_square_vector(vector=vector, M=M)


class Concat(nn.Module):
    """
    Flatten a tuple of inputs.

    Usage:
    ```
    net = nn.Sequential(
        Concat(),
        nn.Linear(3, 1),
    )

    a = Variable(torch.ones(32, 2))
    b = Variable(torch.ones(32, 1))

    result = net((a, b))
    ```
    """
    def __init__(self, *, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


class HuberLoss(nn.Module):
    def __init__(self, delta=1):
        super().__init__()
        self.huber_loss_delta1 = nn.SmoothL1Loss()
        self.delta = delta

    def forward(self, x, x_hat):
        loss = self.huber_loss_delta1(x / self.delta, x_hat / self.delta)
        return loss * self.delta * self.delta


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """
    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


class Detach(nn.Module):
    def __init__(self, wrapped_mlp):
        super().__init__()
        self.wrapped_mlp = wrapped_mlp

    def forward(self, inputs):
        return self.wrapped_mlp.forward(inputs).detach()

    def __getattr__(self, attr_name):
        try:
            return super().__getattr__(attr_name)
        except AttributeError:
            return getattr(self.wrapped_mlp, attr_name)
