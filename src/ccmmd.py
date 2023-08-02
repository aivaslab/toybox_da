"""Module for implementing the class-conditional MMD loss"""
import torch
import torch.nn as nn
import torch.nn.functional as func
import collections
from typing import Optional, Sequence

torch.set_printoptions(precision=2, linewidth=150)


class ClassConditionalMMDLoss(nn.Module):
    """Class definition for Class Conditional MMD Loss"""
    def __init__(self, kernels: Sequence[nn.Module]):
        super(ClassConditionalMMDLoss, self).__init__()
        self.kernels = kernels

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, l_s: torch.Tensor, l_t: torch.Tensor) -> torch.Tensor:
        """Forward method for MMD loss"""
        features = torch.cat([z_s, z_t], dim=0)
        index_matrix = _get_index_matrix(
            src_labels=list(l_s.cpu().numpy()), trgt_labels=list(l_t.cpu().numpy())).to(z_s.device)

        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * index_matrix).sum()  # + 2. / float(batch_size - 1)

        return loss
    
    
class GaussianKernel(nn.Module):
    """Gaussian kernel using convolutions"""

    def __init__(self, alpha: Optional[float] = 1.0):
        super(GaussianKernel, self).__init__()
        self.alpha = alpha

    def forward_old(self, X: torch.Tensor) -> torch.Tensor:
        """Forward method for gaussian kernel"""
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)
        sigma_squared = self.alpha * torch.mean(l2_distance_square.detach())
        return torch.exp(-l2_distance_square / (2 * sigma_squared))
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward method for gaussian kernel"""

        x2 = X ** 2
        x2 = torch.sum(x2, dim=1, keepdim=True).repeat(1, X.shape[0])
        # print(x2, "x2")
        X_reshaped = X.unsqueeze(1)
        # print(X.shape, X_reshaped.shape)

        xp = func.conv1d(input=X_reshaped, weight=X_reshaped).squeeze()
        # print(xp, "xp")
        # print(xp.shape, x2.shape)
        l2_distance_square = func.relu(x2 - 2 * xp + x2.transpose(0, 1))

        sigma_squared = self.alpha * torch.mean(l2_distance_square.detach())
        return torch.exp(-l2_distance_square / (2 * sigma_squared))


def _get_index_matrix(src_labels: list, trgt_labels: list) -> torch.Tensor:
    """
    Compute and return the index matrix
    """
    src_labels_sz = len(src_labels)
    trgt_labels_sz = len(trgt_labels)
    index_matrix = torch.zeros(src_labels_sz + trgt_labels_sz, src_labels_sz + trgt_labels_sz)
    src_labels_ctr = collections.Counter(src_labels)
    trgt_labels_ctr = collections.Counter(trgt_labels)
    
    for i in range(src_labels_sz):
        for j in range(src_labels_sz):
            if src_labels[i] == src_labels[j]:
                index_matrix[i][j] = 1. / float(src_labels_ctr[src_labels[i]] * src_labels_ctr[src_labels[i]])
                
    for i in range(trgt_labels_sz):
        for j in range(trgt_labels_sz):
            if trgt_labels[i] == trgt_labels[j]:
                index_matrix[src_labels_sz + i][src_labels_sz + j] = \
                    1. / float(trgt_labels_ctr[trgt_labels[i]] * trgt_labels_ctr[trgt_labels[i]])
                
    for i in range(src_labels_sz):
        for j in range(trgt_labels_sz):
            if src_labels[i] == trgt_labels[j]:
                index_matrix[i][src_labels_sz + j] = \
                    -1. / float(src_labels_ctr[src_labels[i]] * trgt_labels_ctr[trgt_labels[j]])
                index_matrix[src_labels_sz + i][j] = \
                    -1. / float(src_labels_ctr[src_labels[i]] * trgt_labels_ctr[trgt_labels[j]])
                
    index_matrix = torch.nan_to_num(index_matrix, nan=0.0)
            
    # for i in range(batch_size):
    #     for j in range(batch_size):
    #         if i != j:
    #             index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
    #             index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
    # for i in range(batch_size):
    #     for j in range(batch_size):
    #         index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
    #         index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


def test_index_matrix():
    """Method for testing index matrix"""
    src_labels = [0, 0, 1, 1, 2, 2, 2, 3]
    trgt_labels = [0, 0, 1, 1, 1, 2, 2]
    im = _get_index_matrix(src_labels=src_labels, trgt_labels=trgt_labels)
    print(im)
    
    
def test_gaussian():
    activations = torch.rand(size=(64, 512), dtype=torch.float32)
    # print(activations)
    print(activations.shape)
    gaussian_kernel = GaussianKernel(alpha=1)
    dists = gaussian_kernel.forward_old(activations)
    dists_new = gaussian_kernel.forward(activations)
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            try:
                assert dists[i][j].numpy() - dists_new[i][j].numpy() < 1e-4
            except AssertionError:
                print(i, j, dists[i][j].numpy(), dists_new[i][j].numpy())
    

if __name__ == "__main__":
    # test_index_matrix()
    test_gaussian()