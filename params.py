import torch

# DATA PARAMS
MAX_LENGTH = 512
DATASET = "msmarco_100000"

# MODEL PARAMS
LATENT_SIZE = 3000
HIDDEN_DIM = 1500
BACKBONE_MODEL_ID = "sentence-transformers/msmarco-distilbert-dot-v5"
K_MEANS_LIB = "faiss"

# TRAINING PARAMS
SEED = 1
BATCH_SIZE = 128 * 3
EPOCHS = 10
ELBO_LOSS_ALPHA = 0.0001
REG_LOSS_ALPHA = 5
DIST_LOSS_ALPHA = 30_000
LEARNING_RATE = 1e-4
LOG_EVERY = 20
NUM_CLUSTERS = 25  # for k-means
ANNEAL = True
PROJECT = "Algorithm (iVAE)"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICES = None  # 'auto' is default
KMEANS_FILE = f'/content/drive/MyDrive/ITMO/Аспирантура/Диссертация/Алгоритм/kmeans_n{NUM_CLUSTERS}_{DATASET}_{K_MEANS_LIB}.pickle'
EMBS_FILE = f'/content/drive/MyDrive/ITMO/Аспирантура/Диссертация/Алгоритм/embs_{DATASET}_{DEVICE.type}.pickle'


def print_params():
    import sys
    curr_module = sys.modules[__name__]
    for param in dir(curr_module):
        if param.isupper():
            print(f'{param}:', getattr(curr_module, param))
