import torch


def mahalanobis_score(x, mu, cov_inv):
    diff = x - mu
    return torch.einsum("i,ij,j->", diff, cov_inv, diff)
