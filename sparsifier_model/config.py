from dataclasses import dataclass

import torch


@dataclass
class Config:
    # DATA PARAMS
    max_length: int = 512
    dataset: str = "msmarco_100000"
    # MODEL PARAMS
    latent_dim: int = 3000
    hidden_dim: int = 1500
    backbone_model_id: str = "sentence-transformers/msmarco-distilbert-dot-v5"
    # TRAINING PARAMS
    seed: int = 1
    batch_size: int = 128 * 3
    epochs: int = 10
    elbo_loss_alpha: float = 0.0001
    reg_loss_alpha: float = 5
    dist_loss_alpha: float = 30_000
    learning_rate: float = 1e-4
    log_every: int = 20
    num_clusters: int = 25  # for k-means
    anneal: bool = True
    project: str = "Algorithm (iVAE)"
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    devices = None  # 'auto' is default
    kmeans_file: str = f'/content/drive/MyDrive/ITMO/Аспирантура/Диссертация/Алгоритм/kmeans_n{num_clusters}_{dataset}_faiss.pickle'
    embs_file: str = f'/content/drive/MyDrive/ITMO/Аспирантура/Диссертация/Алгоритм/embs_{dataset}_{device.type}.pickle'
    detect_anomaly: bool = False
