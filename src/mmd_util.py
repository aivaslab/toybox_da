import torch
import torch.nn as nn
from typing import Sequence, Optional

import ot


class EMD1DLoss(nn.Module):
    def __init__(self):
        super(EMD1DLoss, self).__init__()

    @staticmethod
    def forward(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        dist_matrix = torch.dist(z_s.unsqueeze(1), z_t.unsqueeze(0), 2)
        # print(dist_matrix.shape, dist_matrix.requires_grad)
        ot_map = ot.emd_1d(z_s, z_t, metric='euclidean')
        # print(ot_map.shape, ot_map.requires_grad)

        loss = torch.sum(dist_matrix * ot_map)
        # print(loss, loss.requires_grad)

        return loss / z_s.shape[0]


class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):

    def __init__(self, kernels: Sequence[Sequence[nn.Module]], linear: Optional[bool] = True):
        super(JointMultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

        self.thetas = [nn.Identity() for _ in kernels]

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = _update_index_matrix_2(batch_size, self.index_matrix, self.linear).to(z_s[0].device)
        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            feats = sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel
            kernel_matrix *= feats

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


def _update_index_matrix_2(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                           linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            x = 1. / float(batch_size * (batch_size - 1)) * (1 - torch.eye(batch_size))
            y = -1. / float(batch_size * batch_size) * torch.ones(batch_size, batch_size)
            xy, yx = torch.cat((x, y), dim=1), torch.cat((y, x), dim=1)
            index_matrix = torch.cat([xy, yx], dim=0)

    return index_matrix


class IndividualMMDLoss(nn.Module):

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = True):
        super(IndividualMMDLoss, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        batch_size = int(z_s[0].size(-1))
        self.index_matrix = _update_index_matrix_2(batch_size, self.index_matrix, self.linear).to(z_s[0].device)
        self.index_matrix = self.index_matrix.unsqueeze(0).unsqueeze(0)
        kernel_matrix = torch.ones_like(self.index_matrix)
        layer_features = torch.cat([z_s.unsqueeze(1).repeat(1, z_t.shape[0], 1),
                                    z_t.unsqueeze(0).repeat(z_s.shape[0], 1, 1)], dim=-1)
        feats = sum(
            [kernel(layer_features) for kernel in self.kernels])  # Add up the matrix of each kernel
        feats *= kernel_matrix

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        bias_term = 2. / float(batch_size - 1) * torch.ones((z_s.shape[0], z_s.shape[0])).cuda()
        loss = (feats * self.index_matrix).sum((-1, -2))
        loss += bias_term
        return torch.mean(loss)


class IndGaussianKernel(nn.Module):
    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(IndGaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(-1) - X.unsqueeze(-2)) ** 2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


class GaussianKernel(nn.Module):

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # print(X.shape, "gk X")
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)
        # print(l2_distance_square.shape, "gk l2")

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


class JMMDLinear(nn.Module):

    def __init__(self, kernels: Sequence[Sequence[nn.Module]]):
        super(JMMDLinear, self).__init__()
        self.kernels = kernels
        self.index_matrix = None

        self.thetas = [nn.Identity() for _ in kernels]

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = torch.tensor([1.0/float(batch_size)] * (2 * batch_size) + [-1.0/float(batch_size)] *
                                         (2 * batch_size)).cuda()
        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            feats = sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel
            kernel_matrix *= feats

        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        return loss


class GaussianKernelLinear(nn.Module):
    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernelLinear, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        bsize = X.shape[0] // 2
        src_feats, trgt_feats = X[:bsize, :], X[bsize:, :]
        src_feats_rolled, trgt_feats_rolled = torch.roll(src_feats, -1, 0), torch.roll(trgt_feats, -1, 0)
        src_dists = ((src_feats - src_feats_rolled) ** 2).sum(-1)
        trgt_dists = ((trgt_feats - trgt_feats_rolled) ** 2).sum(-1)
        cross_dists_1 = ((src_feats - trgt_feats_rolled) ** 2).sum(-1)
        cross_dists_2 = ((trgt_feats - src_feats_rolled) ** 2).sum(-1)
        l2_distance_square = torch.cat([src_dists, trgt_dists, cross_dists_1, cross_dists_2], dim=0)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))