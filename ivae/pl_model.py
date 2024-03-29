import numpy as np
import torch
import pytorch_lightning as L

from ivae.model import Normal, MLP, weights_init
from loss import ReconstructionLoss, DistanceLoss, FLOPS
from params import (HEAD_MODEL_ID, LEARNING_RATE, INDEP_LOSS_ALPHA,
                    REC_LOSS_ALPHA, DIST_LOSS_ALPHA, REG_LOSS_ALPHA,
                    LOG_EVERY)
from pooling import mean_pooling
from transformers import AutoModel
import wandb


class SparserModel(L.LightningModule):
    def __init__(self, latent_dim, aux_dim, hidden_dim=1000,

                 rec_loss_alpha=REC_LOSS_ALPHA, indep_loss_alpha=INDEP_LOSS_ALPHA,
                 distance_loss_alpha=DIST_LOSS_ALPHA, regularization_loss=FLOPS(alpha=REG_LOSS_ALPHA),

                 prior=None, decoder=None, encoder=None,

                 n_layers=3, activation='lrelu', slope=.1,

                 device='cpu', anneal=False):

        super().__init__()
        head = AutoModel.from_pretrained(HEAD_MODEL_ID).to(device)
        head.eval()
        for p in head.parameters():
            p.requires_grad = False
        self.head = head

        self.data_dim = self.head.config.hidden_size
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.anneal_params = anneal

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # decoder params
        self.f = MLP(latent_dim, self.data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.decoder_var = .01 * torch.ones(1).to(device)
        # encoder params
        self.g = MLP(self.data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(self.data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)
        # losses
        self.reconstruction_loss = ReconstructionLoss(alpha=rec_loss_alpha)
        self.regularization_loss = regularization_loss
        self.distance_loss = DistanceLoss(alpha=distance_loss_alpha)
        self.indep_loss_alpha = indep_loss_alpha

        self.apply(weights_init)

        self._training_hyperparams = [1., 1., 1., 1., 1]
        self.training_step_outputs = []

    def encode(self, token_ids, token_mask):
        x, u = self.__encode_to_x_and_u(token_ids=token_ids, token_mask=token_mask)
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        return z

    def encoder_params(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder_params(self, z):
        f = self.f(z)
        return f, self.decoder_var

    def prior_params(self, u):
        logl = self.logl(u)
        return self.prior_mean, logl.exp()

    def forward(self, x, u):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z, prior_params

    def elbo(self, x, u):
        decoder_params, (g, v), z, prior_params = self.forward(x, u)
        x_rec = decoder_params[0]
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_qz_xu = self.encoder_dist.log_pdf(z, g, v)
        log_pz_u = self.prior_dist.log_pdf(z, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder_dist.log_pdf(z.view(M, 1, self.latent_dim), g.view(1, M, self.latent_dim),
                                                   v.view(1, M, self.latent_dim), reduce=False)
            log_qz = torch.logsumexp(log_qz_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z

        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), x_rec, z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder_var.item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False

    def training_step(self, batch, batch_idx):
        token_ids, token_mask = batch
        x, u = self.__encode_to_x_and_u(token_ids=token_ids, token_mask=token_mask)

        elbo, x_rec, z = self.elbo(x, u)
        rec_loss = self.reconstruction_loss(x, x_rec)
        reg_loss = self.regularization_loss(z)
        dist_loss = self.distance_loss(x, z)
        indep_loss = self.indep_loss_alpha * elbo.mul(-1)
        z_nonzero = torch.count_nonzero(z).float() // len(z)

        loss = rec_loss + reg_loss + dist_loss + indep_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        outs = {
            "loss": loss,
            "rec_loss": rec_loss,
            "reg_loss": reg_loss,
            "dist_loss": dist_loss,
            "indep_loss": indep_loss,
            "z_nonzero": z_nonzero,
        }
        self.training_step_outputs.append(outs)
        if batch_idx % LOG_EVERY == 0:
            wandb.log({'loss': loss})
            wandb.log({'reconstruction loss': rec_loss.detach().clone()})
            wandb.log({'regularization loss': reg_loss.detach().clone()})
            wandb.log({'distance loss': dist_loss.detach().clone()})
            wandb.log({'independence loss': indep_loss.detach().clone()})
            wandb.log({'nonzero count': z_nonzero.detach().clone()})

        return loss

    def on_train_epoch_end(self):
        outs = self.training_step_outputs
        loss = torch.stack([out['loss'] for out in outs]).mean()
        rec_loss = torch.stack([out['rec_loss'] for out in outs]).mean()
        reg_loss = torch.stack([out['reg_loss'] for out in outs]).mean()
        dist_loss = torch.stack([out['dist_loss'] for out in outs]).mean()
        indep_loss = torch.stack([out['indep_loss'] for out in outs]).mean()
        z_nonzero = torch.stack([out['z_nonzero'] for out in outs]).mean()

        print(
            f"loss = {loss:.2f}: " +
            f"reconstruction loss = {rec_loss:.2f}, " +
            f"regularization loss = {reg_loss:.2f}, " +
            f"distance loss = {dist_loss:.2f}, " +
            f"independence loss = {indep_loss:.2f}, " +
            f"nonzero count = {z_nonzero:.2f}"
        )
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
        return optimizer

    def __encode_to_x_and_u(self, token_ids, token_mask):
        x = self.head(input_ids=token_ids, attention_mask=token_mask)
        x = mean_pooling(model_output=x, attention_mask=token_mask)
        u = (token_ids + 1).log()
        return x, u
