import enum
from dataclasses import dataclass, field

import torch


class ModelType(enum.Enum):
    IVAE = 1
    K_SPARSE = 2


@dataclass
class Config:
    model_type: ModelType
    # DATA PARAMS
    max_length: int = 512
    dataset: str = "msmarco_100000"
    # MODEL PARAMS
    ## Common
    latent_dim: int = 3000
    backbone_model_id: str = "sentence-transformers/msmarco-distilbert-dot-v5"
    ## k-sparse specific
    k: int = 600
    ## iVAE specific
    hidden_dim: int = 1500
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
    project: str = field(init=False)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    devices = None  # 'auto' is default
    kmeans_file: str = field(init=False)
    embs_file: str = field(init=False)
    detect_anomaly: bool = False

    def __post_init__(self):
        self.project = (
            "Algorithm (iVAE)"
            if self.model_type == ModelType.IVAE
            else "Algorithm (TopK)"
        )
        base_path = "/content/drive/MyDrive/ITMO/Аспирантура/Диссертация/Алгоритм"
        self.kmeans_file = (
            f"{base_path}/kmeans_n{self.num_clusters}_{self.dataset}_faiss.pickle"
        )
        self.embs_file = f"{base_path}/embs_{self.dataset}_{self.device.type}.pickle"
