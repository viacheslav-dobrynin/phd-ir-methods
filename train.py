import os
import pickle

import numpy as np
import pytorch_lightning as L
import torch
import wandb
from transformers import AutoTokenizer, AutoModel

from dataset import get_dataloader
from ivae.pl_model import SparserModel
from loss import FLOPS
from params import (K_MEANS_LIB, NUM_CLUSTERS, KMEANS_FILE, EMBS_FILE, BACKBONE_MODEL_ID, DEVICE, SEED,
                    REG_LOSS_ALPHA, LATENT_SIZE, HIDDEN_DIM, ELBO_LOSS_ALPHA, DIST_LOSS_ALPHA, ANNEAL, PROJECT, EPOCHS,
                    DEVICES)
from pooling import mean_pooling
from util import create_model_name


def fit_kmeans(embs):
    if K_MEANS_LIB == "fast_pytorch_kmeans":
        import fast_pytorch_kmeans
        kmeans = fast_pytorch_kmeans.KMeans(
            n_clusters=NUM_CLUSTERS,
            init_method="kmeans++",
            max_iter=300,
            minibatch=1000)
        kmeans.fit(embs)
        return kmeans
    elif K_MEANS_LIB == "torch_kmeans":
        import torch_kmeans
        kmeans = torch_kmeans.KMeans(
            n_clusters=NUM_CLUSTERS,
            init_method='k-means++',
            max_iter=300)
        # torch.stack(embs.split(split_size=100)[:-1])
        # kmeans.fit(torch.stack(embs[:3000].split(split_size=100)[:-1]))
        kmeans.fit(embs.unsqueeze(0))
        return kmeans
    elif K_MEANS_LIB == "sklearn":
        import sklearn
        kmeans = sklearn.cluster.KMeans(n_clusters=NUM_CLUSTERS)
        kmeans.fit(embs)
        return kmeans
    else:
        raise ValueError("Unknown k-means lib:", K_MEANS_LIB)


def get_kmeans(dataloader):
    backbone = AutoModel.from_pretrained(BACKBONE_MODEL_ID).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    if os.path.isfile(KMEANS_FILE):
        with open(KMEANS_FILE, 'rb') as f:
            kmeans = pickle.load(f)
            print('Kmeans loaded from drive:', kmeans)
    elif os.path.isfile(EMBS_FILE):
        with open(EMBS_FILE, 'rb') as f:
            embs = pickle.load(f)
            print('Embs loaded from drive:', embs.shape)
            kmeans = fit_kmeans(embs)
            print('Kmeans created from cached embs:', kmeans)
            with open(KMEANS_FILE, 'wb') as f:
                pickle.dump(kmeans, f)
            del embs
    else:
        with torch.no_grad():
            emb_batches = []
            for (token_ids, token_mask) in dataloader:
                token_ids = token_ids.to(DEVICE)
                token_mask = token_mask.to(DEVICE)
                emb_batch = backbone(input_ids=token_ids, attention_mask=token_mask)
                emb_batch = mean_pooling(model_output=emb_batch, attention_mask=token_mask)
                # emb_batches.append(emb_batch.cpu().detach().numpy())
                emb_batches.append(emb_batch)
            # embs = np.concatenate(emb_batches)
            embs = torch.cat(emb_batches)

        print('Embs created:', embs.shape)
        # with open(embs_file, 'wb') as f:
        #   pickle.dump(embs, f)

        kmeans = fit_kmeans(embs)

        print('Kmeans created:', kmeans)

        with open(KMEANS_FILE, 'wb') as f:
            pickle.dump(kmeans, f)

        del embs
        del backbone

    return kmeans


def get_trainer(logger):
    if DEVICES:
        return L.Trainer(max_epochs=EPOCHS, devices=DEVICES, logger=logger)
    else:
        return L.Trainer(max_epochs=EPOCHS, logger=logger)


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID, use_fast=True)
    dataloader, dataset_n, max_iter = get_dataloader(tokenizer)
    kmeans = get_kmeans(dataloader)

    for reg_loss_alpha in REG_LOSS_ALPHA:
        model = SparserModel(latent_dim=LATENT_SIZE, embs_kmeans=kmeans, dataset_n=dataset_n, max_iter=max_iter,
                             hidden_dim=HIDDEN_DIM,

                             elbo_loss_alpha=ELBO_LOSS_ALPHA,
                             distance_loss_alpha=DIST_LOSS_ALPHA,
                             regularization_loss=FLOPS(alpha=reg_loss_alpha),

                             activation='lrelu', device=DEVICE,
                             anneal=ANNEAL)

        desc = ""
        model_name = create_model_name(model=model, desc=desc)
        wandb.init(project=PROJECT, name=model_name)  # mode="disabled"
        wandb_logger = L.loggers.WandbLogger(project=PROJECT, log_model=True, name=model_name, id=model_name)
        trainer = get_trainer(logger=wandb_logger)
        trainer.fit(model=model, train_dataloaders=dataloader)
        wandb.finish()
