import os
import pickle
from typing import Callable, Optional, List, Tuple

import numpy as np
import pytorch_lightning as L
import torch
import wandb
from transformers import AutoTokenizer, AutoModel

from common.pooling import mean_pooling
from params import (NUM_CLUSTERS, KMEANS_FILE, EMBS_FILE, BACKBONE_MODEL_ID, DEVICE, SEED,
                    LATENT_SIZE, HIDDEN_DIM, ANNEAL, PROJECT, EPOCHS,
                    DEVICES, LEARNING_RATE, DATASET, BATCH_SIZE)
from sparsifier_model.ivae.pl_model import SparserModel
from sparsifier_model.util.dataset import get_dataloader
from sparsifier_model.util.model import create_model_name


def fit_kmeans(embs):
    import faiss
    kmeans = faiss.Kmeans(
        d=embs.shape[-1],
        k=NUM_CLUSTERS,
        niter=4,
        gpu=torch.cuda.is_available(),
        verbose=True,
        seed=123)
    kmeans.train(embs.cpu().detach().numpy())
    return kmeans.centroids


def get_kmeans_centroids(dataloader):
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


def get_trainer(logger, detect_anomaly):
    if DEVICES:
        return L.Trainer(max_epochs=EPOCHS, devices=DEVICES, logger=logger, detect_anomaly=detect_anomaly)
    else:
        return L.Trainer(max_epochs=EPOCHS, logger=logger, detect_anomaly=detect_anomaly)


EvalFunctionType = Optional[Callable[[SparserModel], Tuple[List[str], List[List[str]]]]]


def train(elbo_loss_alpha,
          distance_loss_alpha,
          regularization_loss_alpha,
          decoder_var_coef=.01,
          slope=.1,
          eval_fun: EvalFunctionType = None,
          dataset_name=DATASET,
          batch_size=BATCH_SIZE,
          detect_anomaly=False,
          model_desc=""):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID, use_fast=True)
    dataloader, dataset_n, max_iter = get_dataloader(tokenizer=tokenizer,
                                                     dataset_name=dataset_name,
                                                     batch_size=batch_size)
    doc_embs_kmeans_centroids = get_kmeans_centroids(dataloader)

    model = SparserModel(embs_kmeans_centroids=doc_embs_kmeans_centroids,
                         dataset_n=dataset_n, max_iter=max_iter,
                         latent_dim=LATENT_SIZE, hidden_dim=HIDDEN_DIM,

                         elbo_loss_alpha=elbo_loss_alpha,
                         distance_loss_alpha=distance_loss_alpha,
                         regularization_loss_alpha=regularization_loss_alpha,

                         decoder_var_coef=decoder_var_coef,

                         activation='lrelu', slope=slope,
                         device=DEVICE,
                         learning_rate=LEARNING_RATE,
                         anneal=ANNEAL)

    model_name = create_model_name(model=model, desc=model_desc)
    try:
        wandb.init(project=PROJECT, name=model_name)  # mode="disabled"
        wandb_logger = L.loggers.WandbLogger(project=PROJECT, log_model=True, name=model_name, id=model_name)
        trainer = get_trainer(logger=wandb_logger, detect_anomaly=detect_anomaly)
        trainer.fit(model=model, train_dataloaders=dataloader)
        if eval_fun:
            columns, data = eval_fun(model)
            wandb_logger.log_text(key="eval_metrics", columns=columns, data=data)
    finally:
        wandb.finish()
