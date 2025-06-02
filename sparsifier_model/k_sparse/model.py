from typing import Any

import pytorch_lightning as L
from sparsifier_model.config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoModel

from common.pooling import mean_pooling
from sparsifier_model.k_sparse.modules import LN, TiedTranspose, TopK
from sparsifier_model.loss import DistanceLoss


class Autoencoder(L.LightningModule):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self,
        config: Config,
        tied: bool = False,
        normalize: bool = False,
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param activation: activation function
        :param tied: whether to tie the encoder and decoder weights
        """
        super().__init__()
        self.save_hyperparameters()
        backbone = AutoModel.from_pretrained(config.backbone_model_id).to(config.device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone
        n_inputs = backbone.config.hidden_size

        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder: nn.Module = nn.Linear(n_inputs, config.latent_dim, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(config.latent_dim))
        self.activation = TopK(k=config.k)
        if tied:
            self.decoder: nn.Linear | TiedTranspose = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(config.latent_dim, n_inputs, bias=False)
        self.normalize = normalize
        self.learning_rate = config.learning_rate
        self.mse_loss_alpha = config.k_sparse_mse_loss_alpha
        self.distance_loss = DistanceLoss(alpha=config.k_sparse_dist_loss_alpha)
        self.training_step_outputs = []
        self.log_every = config.log_every

        self.stats_last_nonzero: torch.Tensor
        self.latents_activation_frequency: torch.Tensor
        self.latents_mean_square: torch.Tensor
        self.register_buffer(
            "stats_last_nonzero", torch.zeros(config.latent_dim, dtype=torch.long)
        )
        self.register_buffer(
            "latents_activation_frequency",
            torch.ones(config.latent_dim, dtype=torch.float),
        )
        self.register_buffer(
            "latents_mean_square", torch.zeros(config.latent_dim, dtype=torch.float)
        )

    def encode_pre_act(
        self, x: torch.Tensor, latent_slice: slice = slice(None)
    ) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode(self, token_ids, token_mask) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        x = self.backbone(input_ids=token_ids, attention_mask=token_mask)
        x = mean_pooling(model_output=x, attention_mask=token_mask)
        x, _ = self.preprocess(x)
        return self.activation(self.encode_pre_act(x))

    def decode(
        self, latents: torch.Tensor, info: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        # set all indices of self.stats_last_nonzero where (latents != 0) to 0
        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        return latents_pre_act, latents, recons

    def training_step(self, batch, batch_idx):
        token_ids, token_mask = batch
        x = self.backbone(input_ids=token_ids, attention_mask=token_mask)
        x = mean_pooling(model_output=x, attention_mask=token_mask)
        latents_pre_act, latents, recons = self.forward(x)
        mse_loss = self.mse_loss_alpha * F.mse_loss(input=recons, target=x)

        # reg_loss = self.regularization_loss(z)
        dist_loss = self.distance_loss(x, latents)
        loss = mse_loss + dist_loss  # reg_loss + dist_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        outs = {
            "loss": loss,
            # "reg_loss": reg_loss,
            "dist_loss": dist_loss,
        }
        self.training_step_outputs.append(outs)
        if batch_idx % self.log_every == 0:
            wandb.log({"loss": loss})
            # wandb.log({'regularization loss': reg_loss.detach().clone()})
            wandb.log({"distance loss": dist_loss.detach().clone()})

        return loss

    def on_train_epoch_end(self):
        outs = self.training_step_outputs
        loss = torch.stack([out["loss"] for out in outs]).mean()
        # reg_loss = torch.stack([out['reg_loss'] for out in outs]).mean()
        dist_loss = torch.stack([out["dist_loss"] for out in outs]).mean()

        print(
            f"loss = {loss:.2f}: "
            +
            # f"regularization loss = {reg_loss:.2f}, " +
            f"distance loss = {dist_loss:.2f}"
        )
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01
        )
        return optimizer
