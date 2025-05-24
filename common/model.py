import torch
from transformers import AutoModel


def load_model(model_id: str, device: torch.device):
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
