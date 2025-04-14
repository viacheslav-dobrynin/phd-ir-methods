import argparse
import os
import pickle
from collections import defaultdict

import faiss
import numpy as np
import sklearn.cluster
import torch
import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from util.datasets import load_dataset
from params import DEVICE
from spar_k_means_bert.lucene_index import LuceneIndex
from spar_k_means_bert.in_memory_inverted_index import InMemoryInvertedIndex
from util.encode_dense_fun_builder import build_encode_dense_fun


class CorpusDataset(Dataset):

    def __init__(self, corpus):
        self.doc_ids = list(corpus.keys())
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        docs = list(corpus.values())
        self.docs_len = len(docs)
        self.tokenized_docs = tokenize(docs)

    def __len__(self):
        return self.docs_len

    def __getitem__(self, item):
        return self.doc_ids[item], self.tokenized_docs['input_ids'][item], self.tokenized_docs['attention_mask'][item]

    def get_by_doc_id(self, doc_id, device="cpu"):
        idx = self.doc_id_to_idx[doc_id]
        return (self.tokenized_docs['input_ids'][idx].to(device).unsqueeze(0),
                self.tokenized_docs['attention_mask'][idx].to(device).unsqueeze(0))


def load_model():
    model = AutoModel.from_pretrained(args.backbone_model_id).to(DEVICE)
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
    input_ids, _ = dataset.get_by_doc_id(doc_id)
    embs = doc_id_to_embs[doc_id]
    idxs = torch.nonzero(input_ids == token, as_tuple=True)
    return embs[idxs]


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
        for idx, doc_id in enumerate(doc_ids):
            doc_id_to_embs[doc_id] = embs[idx].unsqueeze(0)
    return doc_id_to_embs


def build_hnsw_index():
    hnsw_file_name = f"{args.base_path}hnsw.index"
    faiss_idx_to_token_file_name = f"{args.base_path}faiss_idx_to_token.pickle"

    if args.use_cache and os.path.isfile(hnsw_file_name) and os.path.isfile(faiss_idx_to_token_file_name):
        with open(faiss_idx_to_token_file_name, "rb") as f:
            return faiss.read_index(hnsw_file_name), pickle.load(f)

    if os.path.isfile(hnsw_file_name):
        os.remove(hnsw_file_name)
    if os.path.isfile(faiss_idx_to_token_file_name):
        os.remove(faiss_idx_to_token_file_name)

    token_to_doc_ids = build_token_to_doc_ids()
    hnsw_index = faiss.IndexHNSWFlat(model.config.hidden_size, args.hnsw_M)
    hnsw_index.hnsw.efConstruction = args.hnsw_ef_construction
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
            faiss_idx_to_token[hnsw_index.ntotal - 1] = f"{token}_{i}"

    faiss.write_index(hnsw_index, hnsw_file_name)
    with open(faiss_idx_to_token_file_name, "wb") as f:
        pickle.dump(faiss_idx_to_token, f)
    print(f"Constructed HNSW levels: {np.bincount(faiss.vector_to_array(hnsw_index.hnsw.levels))}")
    return hnsw_index, faiss_idx_to_token


def build_inverted_index():
    if args.in_memory_index:
        inverted_index = InMemoryInvertedIndex(args.base_path, args.use_cache)
    else:
        inverted_index = LuceneIndex(args.base_path, args.use_cache, threshold)
    if inverted_index.size():
        return inverted_index
    for doc_id, contextualized_embs in tqdm.tqdm(iterable=doc_id_to_embs.items(), desc="build_inverted_index"):
        contextualized_embs = contextualized_embs.squeeze(0)
        _, I = hnsw_index.search(contextualized_embs, args.index_n_neighbors)
        assert len(I) == len(contextualized_embs)
        faiss_ids = np.unique(I.flatten()) # this help to remove token repetition
        token_and_cluster_id_list = [faiss_idx_to_token[id] for id in faiss_ids]
        centroids = torch.from_numpy(hnsw_index.reconstruct_batch(faiss_ids))
        scores = torch.max(contextualized_embs @ centroids.T, dim=0).values  # MaxSim
        inverted_index.index(doc_id, token_and_cluster_id_list, scores)
    inverted_index.complete_indexing()
    return inverted_index

def query_tokens_calculator(query):
    tokenized_query = tokenize(query)
    contextualized_embs = encode_to_token_embs(
        input_ids=tokenized_query["input_ids"],
        attention_mask=tokenized_query["attention_mask"])
    contextualized_embs_np = contextualized_embs.squeeze(0).cpu().detach().numpy()
    _, I = hnsw_index.search(contextualized_embs_np, args.search_n_neighbors)
    assert len(I) == len(contextualized_embs_np)
    return list(map(lambda idx: faiss_idx_to_token[idx], np.unique(I.flatten())))


if __name__ == '__main__':
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-mid', '--backbone-model-id', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='backbone model id (default sentence-transformers/all-MiniLM-L6-v2)')
    parser.add_argument('-d', '--dataset', type=str, default='scifact', help='BEIR dataset name (default scifact)')
    parser.add_argument('-l', '--dataset-length', type=int, default=None, help='Dataset length (default None, all dataset)')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='batch size (default 128)')
    parser.add_argument('-kmn', '--kmeans-n-clusters', type=int, default=8, help='kmeans clusters number (default 8)')
    parser.add_argument('-M', '--hnsw-M', type=int, default=32, help='the number of neighbors used in the graph. A larger M is more accurate but uses more memory (default 32)')
    parser.add_argument('-efs', '--hnsw-ef-search', type=int, default=16, help='HNSW ef search param (default 16)')
    parser.add_argument('-efc', '--hnsw-ef-construction', type=int, default=40, help='HNSW ef construction param (default 40)')
    parser.add_argument('-in', '--index-n-neighbors', type=int, default=8, help='index neighbors number (default 8)')
    parser.add_argument('-stk', '--search-top-k', type=int, default=1000, help='search tok k results (default 1000)')
    parser.add_argument('-sn', '--search-n-neighbors', type=int, default=3, help='search neighbors number (default 3)')
    parser.add_argument('--train-hnsw-only', action="store_true", help='train hnsw only (default False)')
    parser.add_argument('-imi', '--in-memory-index', action="store_true", help='in-memory inverted index type (default False)')
    parser.add_argument('-c', '--use-cache', action="store_true", help='use cache (default False)')
    parser.add_argument('-p', '--base-path', type=str, default='./', help='base path (default ./)')
    args = parser.parse_args()
    print(f"Params: {args}")
    # Data, tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_id, use_fast=True)
    corpus, queries, qrels = load_dataset(dataset=args.dataset, length=args.dataset_length)
    sep = " "
    corpus = {doc_id: (doc["title"] + sep + doc["text"]).strip() for doc_id, doc in corpus.items()}
    print(f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}")
    dataset = CorpusDataset(corpus)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)
    model = load_model()
    encode_dense = build_encode_dense_fun(tokenizer=tokenizer, model=model)
    threshold = 0.8 * encode_dense("She enjoys reading books in her free time.") @ encode_dense("In her leisure hours, she likes to read novels.").T
    threshold = threshold.squeeze(0).cpu()
    print(f"Dense similarity threshold: {threshold}")
    # Indexing
    doc_id_to_embs = build_doc_id_to_embs()
    hnsw_index, faiss_idx_to_token = build_hnsw_index()
    print("HNSW index size: ", hnsw_index.ntotal)
    if args.train_hnsw_only:
        print("HNSW index is trained")
        exit(0) # TODO: extract to function and use return
    hnsw_index.hnsw.efSearch = args.hnsw_ef_search
    inverted_index = build_inverted_index()
    # Retrieval
    results = inverted_index.search(queries, query_tokens_calculator, args.search_top_k)
    retriever = EvaluateRetrieval(score_function="dot")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    print(ndcg, _map, recall, precision, mrr)
    with open("retrieval_results.txt", "a") as f:
        f.write(str(args))
        f.write("\n")
        f.write(str(ndcg))
        f.write(str(mrr))
        f.write(str(_map))
        f.write(str(recall))
        f.write(str(precision))
        f.write("\n\n")
