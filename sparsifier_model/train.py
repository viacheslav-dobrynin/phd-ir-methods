import enum
import os
import pickle
from typing import Callable, Optional, List, Tuple

import numpy as np
import pytorch_lightning as L
from sparsifier_model.k_sparse.model import Autoencoder
import torch
import wandb
from transformers import AutoTokenizer, AutoModel

from common.pooling import mean_pooling
from sparsifier_model.config import Config, ModelType
from sparsifier_model.ivae.pl_model import SparserModel
from sparsifier_model.util.dataset import get_dataloader
from sparsifier_model.util.model import create_model_name


def fit_kmeans(config: Config, embs):
    import faiss

    kmeans = faiss.Kmeans(
        d=embs.shape[-1],
        k=config.num_clusters,
        niter=4,
        gpu=torch.cuda.is_available(),
        verbose=True,
        seed=config.seed,
    )
    kmeans.train(embs.cpu().detach().numpy())
    return kmeans.centroids


def get_kmeans_centroids(config: Config, dataloader):
    backbone = AutoModel.from_pretrained(config.backbone_model_id).to(config.device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    if os.path.isfile(config.kmeans_file):
        with open(config.kmeans_file, "rb") as f:
            kmeans = pickle.load(f)
            print("Kmeans loaded from drive:", kmeans)
    elif os.path.isfile(config.embs_file):
        with open(config.embs_file, "rb") as f:
            embs = pickle.load(f)
            print("Embs loaded from drive:", embs.shape)
            kmeans = fit_kmeans(config, embs)
            print("Kmeans created from cached embs:", kmeans)
            with open(config.kmeans_file, "wb") as f:
                pickle.dump(kmeans, f)
            del embs
    else:
        with torch.no_grad():
            emb_batches = []
            for token_ids, token_mask in dataloader:
                token_ids = token_ids.to(config.device)
                token_mask = token_mask.to(config.device)
                emb_batch = backbone(input_ids=token_ids, attention_mask=token_mask)
                emb_batch = mean_pooling(
                    model_output=emb_batch, attention_mask=token_mask
                )
                # emb_batches.append(emb_batch.cpu().detach().numpy())
                emb_batches.append(emb_batch)
            # embs = np.concatenate(emb_batches)
            embs = torch.cat(emb_batches)

        print("Embs created:", embs.shape)
        # with open(embs_file, 'wb') as f:
        #   pickle.dump(embs, f)

        kmeans = fit_kmeans(config, embs)

        print("Kmeans created:", kmeans)

        with open(config.kmeans_file, "wb") as f:
            pickle.dump(kmeans, f)

        del embs
        del backbone

    return kmeans


def get_trainer(config: Config, logger):
    if config.devices:
        return L.Trainer(
            max_epochs=config.epochs,
            devices=config.devices,
            logger=logger,
            detect_anomaly=config.detect_anomaly,
        )
    else:
        return L.Trainer(
            max_epochs=config.epochs,
            logger=logger,
            detect_anomaly=config.detect_anomaly,
        )


EvalFunctionType = Optional[
    Callable[[SparserModel | Autoencoder], Tuple[List[str], List[List[str]]]]
]


def train(
    config: Config,
    decoder_var_coef=0.01,
    slope=0.1,
    eval_fun: EvalFunctionType = None,
    model_desc="",
):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.backbone_model_id, use_fast=True)
    dataloader, dataset_n, max_iter = get_dataloader(config=config, tokenizer=tokenizer)

    match config.model_type:
        case ModelType.IVAE:
            doc_embs_kmeans_centroids = get_kmeans_centroids(
                config=config, dataloader=dataloader
            )
            model = SparserModel(
                config=config,
                embs_kmeans_centroids=doc_embs_kmeans_centroids,
                dataset_n=dataset_n,
                max_iter=max_iter,
                decoder_var_coef=decoder_var_coef,
                activation="lrelu",
                slope=slope,
            )
        case ModelType.K_SPARSE:
            model = Autoencoder(config=config)

    model_name = create_model_name(model=model, desc=model_desc)
    try:
        wandb.init(project=config.project, name=model_name)  # mode="disabled"
        wandb_logger = L.loggers.WandbLogger(
            project=config.project, log_model=True, name=model_name, id=model_name
        )
        trainer = get_trainer(config=config, logger=wandb_logger)
        trainer.fit(model=model, train_dataloaders=dataloader)
        if eval_fun:
            columns, data = eval_fun(model)
            wandb_logger.log_text(key="eval_metrics", columns=columns, data=data)
    finally:
        wandb.finish()
