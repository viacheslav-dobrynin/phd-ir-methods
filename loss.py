from params import REC_LOSS
import torch.nn.functional as F


class ReconstructionLoss:
    def __init__(self, alpha=REC_LOSS):
        self.alpha = alpha

    def __call__(self, x, x_rec):
        return self.alpha * F.mse_loss(x, x_rec)
