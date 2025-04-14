import torch
import torch.nn.functional as F


class ReconstructionLoss:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x, x_rec):
        return self.alpha * F.mse_loss(x, x_rec)


class DistanceLoss:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x, x_rep):
        x_dots = self.__dot_products(x)
        x_relative_dots = x_dots / torch.diagonal(x_dots).unsqueeze(0).T

        x_rep_dots = self.__dot_products(x_rep)
        x_rep_relative_dots = x_rep_dots / torch.diagonal(x_rep_dots).unsqueeze(0).T

        return self.alpha * torch.linalg.matrix_norm(x_relative_dots - x_rep_relative_dots, ord='fro') / x.shape[0]

    def __dot_products(self, x):
        _orig_dtype = x.dtype
        x = x.to(torch.float64)
        dot_products = x.mm(x.T).to(_orig_dtype)
        return dot_products


class L1:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x_rep):
        all_params = torch.cat([t.view(-1) for t in x_rep])
        return self.alpha * torch.norm(all_params, 1)


class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x_rep):
        return self.alpha * torch.sum(torch.mean(torch.abs(x_rep), dim=0) ** 2)
