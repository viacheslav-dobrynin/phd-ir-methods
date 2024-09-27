from collections import defaultdict

import faiss
import numpy as np
import sklearn.cluster
import torch
import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from dataset import load_dataset
from params import BACKBONE_MODEL_ID, DEVICE


class CorpusDataset(Dataset):

    def __init__(self, corpus):
        self.doc_ids = list(corpus.keys())
        docs = list(corpus.values())
        self.docs_len = len(docs)
        self.tokenized_docs = tokenize(docs).to(DEVICE)

    def __len__(self):
        return self.docs_len

    def __getitem__(self, item):
        return self.doc_ids[item], self.tokenized_docs['input_ids'][item], self.tokenized_docs['attention_mask'][item]


def load_model():
    model = AutoModel.from_pretrained(BACKBONE_MODEL_ID).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Encode text
def encode_to_token_embs(input_ids, attention_mask):
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    return model_output.last_hidden_state


def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)


def get_contextualized_embs(token, doc_id):
    input_ids, _, embs = doc_id_to_embs[doc_id]
    idxs = torch.nonzero(input_ids == token, as_tuple=True)
    return embs[idxs]


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)

    def add(self, token_and_cluster_id, doc_id_and_score):
        self.index[token_and_cluster_id].add(doc_id_and_score)

    def search(self, query, top_k, n_neighbors=3):
        contextualized_embs = encode_to_token_embs(**tokenize(query))
        doc_id_and_score_list = []
        for contextualized_emb in contextualized_embs.squeeze(0):
            _, I = index.search(np.array([contextualized_emb.cpu().detach().numpy()]), n_neighbors)
            token_and_cluster_id_list = [faiss_idx_to_token[id] for id in I[0].tolist()]
            for token_and_cluster_id in token_and_cluster_id_list:
                for doc_id_and_score in self.index[token_and_cluster_id]:
                    doc_id_and_score_list.append(doc_id_and_score)

        doc_id_to_score = defaultdict(float)
        for doc_id, score in doc_id_and_score_list:
            doc_id_to_score[doc_id] += score
        return sorted(doc_id_to_score.items(), key=lambda e: e[1], reverse=True)[:top_k]


if __name__ == '__main__':
    corpus, queries, qrels = load_dataset()
    sep = " "
    corpus = {doc_id: (doc["title"] + sep + doc["text"]).strip() for doc_id, doc in corpus.items()}
    print(f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID, use_fast=True)
    dataset = CorpusDataset(corpus)
    batch_size = 128
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    model = load_model()

    token_to_doc_ids = defaultdict(set)
    skip_tokens = {0, 101, 102}

    for doc_ids, token_ids_batch, _ in tqdm.tqdm(iterable=dataloader, desc="build_token_to_doc_ids"):
        for idx, doc_id in enumerate(doc_ids):
            for token_id in token_ids_batch[idx]:
                token_id = token_id.item()
                if token_id not in skip_tokens:
                    token_to_doc_ids[token_id].add(doc_id)

    doc_id_to_embs = {}
    for doc_ids, token_ids_batch, attention_mask in tqdm.tqdm(iterable=dataloader, desc="encode_to_token_embs"):
        embs = encode_to_token_embs(input_ids=token_ids_batch, attention_mask=attention_mask)
        embs = embs.cpu()
        for idx, doc_id in enumerate(doc_ids):
            doc_id_to_embs[doc_id] = (token_ids_batch[idx], attention_mask[idx], embs[idx])

    n_clusters = 8
    M = 32  # is the number of neighbors used in the graph. A larger M is more accurate but uses more memory
    faiss_idx_to_token = {}
    index = faiss.IndexHNSWFlat(model.config.hidden_size, M)

    for token, doc_ids in tqdm.tqdm(iterable=token_to_doc_ids.items(), desc="build_hnsw"):
        emb_batches = []
        for doc_id in doc_ids:
            emb_batch = get_contextualized_embs(token, doc_id)
            emb_batches.append(emb_batch.cpu().detach().numpy())
        embs = np.concatenate(emb_batches)
        kmeans = sklearn.cluster.KMeans(
            n_clusters=n_clusters if len(embs) > n_clusters else len(embs),
            init='k-means++',
            n_init='auto')
        kmeans.fit(embs)
        for i, centroid in enumerate(kmeans.cluster_centers_):
            index.add(np.array([centroid]))
            faiss_idx_to_token[index.ntotal - 1] = (token, i)

    index_n_neighbors = 8
    inverted_index = InvertedIndex()

    for token, doc_ids in tqdm.tqdm(iterable=token_to_doc_ids.items(), desc="build_inverted_index"):
        for doc_id in doc_ids:
            input_ids, attention_mask, embs = doc_id_to_embs[doc_id]
            idxs = torch.nonzero(input_ids == token, as_tuple=True)
            contextualized_embs = embs[idxs]
            doc_emb = mean_pooling(embs, attention_mask).cpu().detach().numpy()
            for contextualized_emb in contextualized_embs:
                D, I = index.search(np.array([contextualized_emb.cpu().detach().numpy()]), index_n_neighbors)
                centroids = [index.reconstruct(id) for id in I[0].tolist()]
                token_and_cluster_id_list = [faiss_idx_to_token[id] for id in I[0].tolist()]
                centroids = np.stack(centroids)
                scores = np.squeeze(doc_emb @ np.stack(centroids).T, 0)
                for token_and_cluster_id, score in zip(token_and_cluster_id_list, scores):
                    inverted_index.add(token_and_cluster_id, (doc_id, score))

    results = {}
    top_k = 1000
    search_n_neighbors = 3
    query_ids = list(queries.keys())
    for query_id in tqdm.tqdm(iterable=query_ids, desc="search"):
        doc_id_and_score_list = inverted_index.search(query=queries[query_id],
                                                      top_k=top_k,
                                                      n_neighbors=search_n_neighbors)
        query_result = {}
        for doc_id, score in doc_id_and_score_list:
            query_result[doc_id] = score
        results[query_id] = query_result

    retriever = EvaluateRetrieval(score_function="dot")
    retrieval_result = retriever.evaluate(qrels, results, retriever.k_values)
    print(retrieval_result)
