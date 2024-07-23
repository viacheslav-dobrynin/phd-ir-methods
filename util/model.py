import torch
import torch.nn.functional as F

from params import DEVICE, MAX_LENGTH
from pooling import mean_pooling


def create_model_name(model, desc=""):
    return ("iVAE_s" + str(model.latent_dim) +
            "_elbo_" + str(model.elbo_loss_alpha) +
            f"_{model.regularization_loss.__class__.__name__}_{str(model.regularization_loss.alpha)}" +
            "_dist_" + str(model.distance_loss.alpha) +
            "_n_clusts_" + str(model.embs_kmeans.n_clusters) +
            "_slope_" + str(model.slope) +
            desc)


def build_encode_sparse_fun(tokenizer, model, threshold, zeroing_type="quantile"):
    def encode_sparse_from_tokens(token_ids, token_mask):
        with torch.no_grad():
            z = model.encode(token_ids=token_ids, token_mask=token_mask)
            if threshold is not None:
                if zeroing_type == "quantile":
                    q = torch.quantile(z, torch.tensor([threshold, 1.0 - threshold]).to(DEVICE), dim=1, keepdim=True)
                    z = torch.where((z <= q[0]) | (z >= q[1]), z, 0.0)
                elif zeroing_type == "threshold":
                    z[torch.abs(z) < threshold] = 0
                else:
                    raise ValueError(f"Zeroing type '{zeroing_type}' not supported")
        return z

    def encode_sparse_from_docs(docs: list[str]):
        tokenized = tokenizer(docs,
                              return_tensors="pt",
                              padding='max_length',
                              truncation=True,
                              max_length=MAX_LENGTH).to(DEVICE)
        return encode_sparse_from_tokens(token_ids=tokenized["input_ids"], token_mask=tokenized["attention_mask"])

    return encode_sparse_from_docs if tokenizer else encode_sparse_from_tokens


def build_encode_dense_fun(tokenizer, model):
    def encode_dense(docs):
        # Tokenize sentences
        encoded_input = tokenizer(docs, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input, return_dict=True)
        # Perform pooling
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings

    return encode_dense


def sparsify_abs(x, sparse_ratio=0.2):
    k = int(sparse_ratio * x.shape[1])
    absx = torch.abs(x)
    topval = absx.topk(k, dim=1)[0][:, -1]
    topval = topval.expand(absx.shape[1], absx.shape[0]).permute(1, 0)
    return (torch.sign(x) * F.relu(absx - topval)).to(x)
