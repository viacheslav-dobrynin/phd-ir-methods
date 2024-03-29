import torch.nn.functional as F


class ReconstructionLoss:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x, x_rec):
        return self.alpha * F.mse_loss(x, x_rec)
