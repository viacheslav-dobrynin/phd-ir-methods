import argparse
import pickle
from collections import defaultdict

import faiss
import numpy as np
import sklearn.cluster
import torch
import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval
from faiss import read_index, write_index
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
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
def mean_pooling(token_embeddings, attention_mask):
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
            _, I = hnsw_index.search(np.array([contextualized_emb.cpu().detach().numpy()]), n_neighbors)
            token_and_cluster_id_list = [faiss_idx_to_token[id] for id in I[0].tolist()]
            for token_and_cluster_id in token_and_cluster_id_list:
                for doc_id_and_score in self.index[token_and_cluster_id]:
                    doc_id_and_score_list.append(doc_id_and_score)

        doc_id_to_score = defaultdict(float)
        for doc_id, score in doc_id_and_score_list:
            doc_id_to_score[doc_id] += score
        return sorted(doc_id_to_score.items(), key=lambda e: e[1], reverse=True)[:top_k]


def build_token_to_doc_ids():
    token_to_doc_ids = defaultdict(set)
    skip_tokens = {0, 101, 102}
    for doc_ids, token_ids_batch, _ in tqdm.tqdm(iterable=dataloader, desc="build_token_to_doc_ids"):
        for idx, doc_id in enumerate(doc_ids):
            for token_id in token_ids_batch[idx]:
                token_id = token_id.item()
                if token_id not in skip_tokens:
                    token_to_doc_ids[token_id].add(doc_id)
    return token_to_doc_ids


def build_doc_id_to_embs():
    doc_id_to_embs = {}
    for doc_ids, token_ids_batch, attention_mask in tqdm.tqdm(iterable=dataloader, desc="encode_to_token_embs"):
        embs = encode_to_token_embs(input_ids=token_ids_batch, attention_mask=attention_mask)
        embs = embs.cpu()
        token_ids_batch = token_ids_batch.cpu()
        attention_mask = attention_mask.cpu()
        for idx, doc_id in enumerate(doc_ids):
            doc_id_to_embs[doc_id] = (
            token_ids_batch[idx].unsqueeze(0), attention_mask[idx].unsqueeze(0), embs[idx].unsqueeze(0))
    return doc_id_to_embs


def build_hnsw_index():
    base_path = "./"
    hnsw_file_name = f"{base_path}hnsw.index"
    faiss_idx_to_token_file_name = f"{base_path}faiss_idx_to_token.pickle"

    if os.path.isfile(hnsw_file_name) and os.path.isfile(faiss_idx_to_token_file_name):
        with open(faiss_idx_to_token_file_name, "rb") as f:
            return read_index(hnsw_file_name), pickle.load(f)

    if os.path.isfile(hnsw_file_name):
        os.remove(hnsw_file_name)
    if os.path.isfile(faiss_idx_to_token_file_name):
        os.remove(faiss_idx_to_token_file_name)

    hnsw_index = faiss.IndexHNSWFlat(model.config.hidden_size, args.hnsw_M)
    faiss_idx_to_token = {}

    for token, doc_ids in tqdm.tqdm(iterable=token_to_doc_ids.items(), desc="build_hnsw"):
        emb_batches = []
        for doc_id in doc_ids:
            emb_batch = get_contextualized_embs(token, doc_id)
            emb_batches.append(emb_batch.cpu().detach().numpy())
        embs = np.concatenate(emb_batches)
        kmeans = sklearn.cluster.KMeans(
            n_clusters=args.kmeans_n_clusters if len(embs) > args.kmeans_n_clusters else len(embs),
            init='k-means++',
            n_init='auto')
        kmeans.fit(embs)
        for i, centroid in enumerate(kmeans.cluster_centers_):
            hnsw_index.add(np.array([centroid]))
            faiss_idx_to_token[hnsw_index.ntotal - 1] = (token, i)

    write_index(hnsw_index, hnsw_file_name)
    with open(faiss_idx_to_token_file_name, "wb") as f:
        pickle.dump(faiss_idx_to_token, f)

    return hnsw_index, faiss_idx_to_token


def build_inverted_index():
    inverted_index = InvertedIndex()
    for token, doc_ids in tqdm.tqdm(iterable=token_to_doc_ids.items(), desc="build_inverted_index"):
        contextualized_embs_list = []
        doc_ids_list = []
        doc_id_to_doc_emb = {}
        for doc_id in doc_ids:
            input_ids, attention_mask, embs = doc_id_to_embs[doc_id]
            idxs = torch.nonzero(input_ids == token, as_tuple=True)
            assert idxs[1].numel() != 0
            contextualized_embs = embs[idxs]
            contextualized_embs_list.append(contextualized_embs)
            doc_ids_list.extend([doc_id] * contextualized_embs.shape[0])
            doc_id_to_doc_emb[doc_id] = mean_pooling(embs, attention_mask).cpu().detach().numpy()

        assert len(contextualized_embs_list) != 0

        contextualized_embs = torch.cat(contextualized_embs_list, dim=0).cpu().detach().numpy()

        _, I = hnsw_index.search(contextualized_embs, args.index_n_neighbors)
        assert len(I) == len(contextualized_embs)
        for idx in range(len(contextualized_embs)):
            doc_id = doc_ids_list[idx]
            token_and_cluster_id_list = [faiss_idx_to_token[id] for id in I[idx]]
            doc_emb = doc_id_to_doc_emb[doc_id]
            centroids = hnsw_index.reconstruct_batch(I[idx])
            scores = np.squeeze(doc_emb @ centroids.T, 0)
            assert len(token_and_cluster_id_list) == len(scores)
            for token_and_cluster_id, score in zip(token_and_cluster_id_list, scores):
                inverted_index.add(token_and_cluster_id, (doc_id, score))
    return inverted_index


def perform_searches():
    results = {}
    query_ids = list(queries.keys())
    for query_id in tqdm.tqdm(iterable=query_ids, desc="search"):
        doc_id_and_score_list = inverted_index.search(query=queries[query_id],
                                                      top_k=args.search_top_k,
                                                      n_neighbors=args.search_n_neighbors)
        query_result = {}
        for doc_id, score in doc_id_and_score_list:
            query_result[doc_id] = score
        results[query_id] = query_result
    return results


if __name__ == '__main__':
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='scifact', help='BEIR dataset name (default scifact)')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='batch size (default 128)')
    parser.add_argument('-kmn', '--kmeans-n-clusters', type=int, default=8, help='kmeans clusters number (default 8)')
    parser.add_argument('-M', '--hnsw-M', type=int, default=32, help='the number of neighbors used in the graph. A larger M is more accurate but uses more memory (default 32)')
    parser.add_argument('-in', '--index-n-neighbors', type=int, default=8, help='index neighbors number (default 8)')
    parser.add_argument('-stk', '--search-top-k', type=int, default=1000, help='search tok k results (default 1000)')
    parser.add_argument('-sn', '--search-n-neighbors', type=int, default=3, help='search neighbors number (default 3)')
    args = parser.parse_args()
    print(f"Params: {args}")
    # Data, tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID, use_fast=True)
    corpus, queries, qrels = load_dataset(dataset=args.dataset)
    sep = " "
    corpus = {doc_id: (doc["title"] + sep + doc["text"]).strip() for doc_id, doc in corpus.items()}
    print(f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}")
    dataset = CorpusDataset(corpus)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)
    model = load_model()
    # Indexing
    token_to_doc_ids = build_token_to_doc_ids()
    doc_id_to_embs = build_doc_id_to_embs()
    hnsw_index, faiss_idx_to_token = build_hnsw_index()
    inverted_index = build_inverted_index()
    # Retrieval
    results = perform_searches()
    retriever = EvaluateRetrieval(score_function="dot")
    retrieval_result = retriever.evaluate(qrels, results, retriever.k_values)
    print(retrieval_result)
    with open("retrieval_results.txt", "a") as f:
        f.write(str(args))
        f.write("\n")
        f.write(str(retrieval_result))
        f.write("\n\n")
