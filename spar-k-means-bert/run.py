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
from transformers import AutoTokenizer, AutoModel, BatchEncoding
import os
from dataset import load_dataset
from params import BACKBONE_MODEL_ID, DEVICE


class CorpusDataset(Dataset):

    def __init__(self, corpus):
        self.doc_ids = list(corpus.keys())
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        docs = list(corpus.values())
        self.docs_len = len(docs)
        self.tokenized_docs = tokenize(docs).to(DEVICE)

    def __len__(self):
        return self.docs_len

    def __getitem__(self, item):
        return self.doc_ids[item], self.tokenized_docs['input_ids'][item], self.tokenized_docs['attention_mask'][item]

    def get_by_doc_id(self, doc_id, device=DEVICE) -> tuple[BatchEncoding, BatchEncoding]:
        idx = self.doc_id_to_idx[doc_id]
        return (self.tokenized_docs['input_ids'][idx].to(device).unsqueeze(0),
                self.tokenized_docs['attention_mask'][idx].to(device).unsqueeze(0))

class PQDocumentEmbeddings:
    def __init__(self, model, dataset: CorpusDataset, dataloader: DataLoader):
        self.doc_id_to_index: dict[str, faiss.IndexPQ] = {}
        self.dataset = dataset

        for doc_ids, token_ids_batch, attention_mask in tqdm.tqdm(iterable=dataloader, desc="encode_to_token_embs"):
            embs = encode_to_token_embs(model, input_ids=token_ids_batch, attention_mask=attention_mask)
            embs = embs.cpu().detach()

            for idx, doc_id in enumerate(doc_ids):
                pq = faiss.IndexPQ(model.config.hidden_size, 8, 8)
                
                #we train quantization on whole batch for more optimal quantization
                pq.train(embs.numpy().astype(np.float32))

                #then save only doc-related embeddings
                pq.add(embs[idx].unsqueeze(0).numpy().astype(np.float32))

                self.doc_id_to_index[doc_id] = pq

    def get_by_document(self, doc_id: str) -> list[torch.Tensor]:
        index = self.doc_id_to_index[doc_id]
        reconstructed_embedding = index.reconstruct_n(0, index.ntotal)
        return torch.tensor(reconstructed_embedding)
    
    def get_by_token(self, doc_id: str, token: int) -> tuple[list[torch.Tensor], BatchEncoding]:
        input_ids, attention_mask = self.dataset.get_by_doc_id(doc_id)
        index = self.doc_id_to_index[doc_id]
        reconstructed_embedding = torch.tensor(index.reconstruct_n(0, index.ntotal))
        idxs = torch.nonzero(input_ids == token, as_tuple=True)
        return reconstructed_embedding[idxs], attention_mask
    
class DocumentEmbeddings:
    def __init__(self, model, dataset: CorpusDataset, dataloader: DataLoader):
        self.doc_id_to_embs: dict[str, faiss.IndexPQ] = {}
        self.dataset = dataset

        for doc_ids, token_ids_batch, attention_mask in tqdm.tqdm(iterable=dataloader, desc="encode_to_token_embs"):
            embs = encode_to_token_embs(model, input_ids=token_ids_batch, attention_mask=attention_mask)
            embs = embs.cpu().detach()
            for idx, doc_id in enumerate(doc_ids):
                self.doc_id_to_embs[doc_id] = embs[idx].unsqueeze(0)

    def get_by_document(self, doc_id: str) -> list[torch.Tensor]:
        return self.doc_id_to_embs[doc_id]
    
    def get_by_token(self, doc_id: str, token: int) -> tuple[list[torch.Tensor], BatchEncoding]:
        input_ids, attention = dataset.get_by_doc_id(doc_id)
        embs = self.doc_id_to_embs[doc_id]
        idxs = torch.nonzero(input_ids == token, as_tuple=True)
        return embs[idxs], attention

class VectorIndex:
    def __init__(self, args, model):

        if args.use_pq:
            self.hnsw_index = faiss.IndexHNSWSQ(model.config.hidden_size, faiss.ScalarQuantizer.QT_8bit, args.hnsw_M)
            self.hnsw_index.is_trained
        else:
            self.hnsw_index = faiss.IndexHNSWFlat(model.config.hidden_size, args.hnsw_M)

        self.faiss_idx_to_token = {}

    def build(self, documentEmbeddings: DocumentEmbeddings, token_to_doc_ids: defaultdict[int, set[str]]):
        for token, doc_ids in tqdm.tqdm(iterable=token_to_doc_ids.items(), desc="build_hnsw"):
            emb_batches = []
            for doc_id in doc_ids:
                emb_batch, _ = documentEmbeddings.get_by_token(doc_id, token)
                emb_batches.append(emb_batch.cpu().detach().numpy())
            embs = np.concatenate(emb_batches)

            n_clusters=args.kmeans_n_clusters if len(embs) > args.kmeans_n_clusters else len(embs)
            kmeans = faiss.Kmeans(d=embs.shape[1], k=n_clusters, niter=20, verbose=False)
            kmeans.train(embs)

            #TODO fix training
            if self.hnsw_index.is_trained is not None and not self.hnsw_index.is_trained:
                self.hnsw_index.train(embs)

            for i, centroid in enumerate(kmeans.centroids):
                self.hnsw_index.add(np.array([centroid]))
                self.faiss_idx_to_token[self.hnsw_index.ntotal - 1] = (token, i)



    def load(self, args):
        hnsw_file_name = self._get_index_filename(args.base_path)
        faiss_idx_to_token_file_name = self._get_mapping_filename(args.base_path)

        self.hnsw_index: faiss.IndexHNSWSQ = read_index(hnsw_file_name)

        with open(faiss_idx_to_token_file_name, "rb") as f:
            self.faiss_idx_to_token = pickle.load(f)

        return self

    def store(self, args):
        hnsw_file_name = self._get_index_filename(args.base_path)
        faiss_idx_to_token_file_name = self._get_mapping_filename(args.base_path)

        if os.path.isfile(hnsw_file_name):
            os.remove(hnsw_file_name)
        if os.path.isfile(faiss_idx_to_token_file_name):
            os.remove(faiss_idx_to_token_file_name)

        write_index(self.hnsw_index, hnsw_file_name)
        with open(faiss_idx_to_token_file_name, "wb") as f:
            pickle.dump(self.faiss_idx_to_token, f)

    def search(self, contextualized_embs, n_neighbors):
        return self.hnsw_index.search(contextualized_embs, n_neighbors)
    
    def map_ids_to_token_ids(self, ids):
        return [self.faiss_idx_to_token[id] for id in ids]

    def reconstruct(self, ids):
        return self.hnsw_index.reconstruct_batch(ids)

    def _get_index_filename(self, filename: str) -> str:
        return f"{filename}hnsw.index"

    def _get_mapping_filename(self, filename: str) -> str:
        return f"{filename}faiss_idx_to_token.pickle"
    
def load_model() -> AutoModel:
    model = AutoModel.from_pretrained(BACKBONE_MODEL_ID).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Encode text
def encode_to_token_embs(model: AutoModel, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    return model_output.last_hidden_state


def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)



class InvertedIndex:
    def __init__(self, model, vector_index: VectorIndex):
        self.index = defaultdict(set)
        self.model = model 
        self.vector_index: VectorIndex = vector_index

    def add(self, token_and_cluster_id, doc_id_and_score):
        self.index[token_and_cluster_id].add(doc_id_and_score)

    def search(self, query, top_k, n_neighbors=3):
        contextualized_embs = encode_to_token_embs(self.model, **tokenize(query))
        doc_id_and_score_list = []
        for contextualized_emb in contextualized_embs.squeeze(0):
            _, I = vector_index.search(np.array([contextualized_emb.cpu().detach().numpy()]), n_neighbors)
            token_and_cluster_id_list = vector_index.map_ids_to_token_ids(I[0].tolist())
            for token_and_cluster_id in token_and_cluster_id_list:
                for doc_id_and_score in self.index[token_and_cluster_id]:
                    doc_id_and_score_list.append(doc_id_and_score)

        doc_id_to_score = defaultdict(float)
        for doc_id, score in doc_id_and_score_list:
            doc_id_to_score[doc_id] += score
        return sorted(doc_id_to_score.items(), key=lambda e: e[1], reverse=True)[:top_k]


def build_token_to_doc_ids(dataloader: DataLoader) -> defaultdict[int, set[str]]:
    token_to_doc_ids = defaultdict(set)
    skip_tokens = {0, 101, 102}
    for doc_ids, token_ids_batch, _ in tqdm.tqdm(iterable=dataloader, desc="build_token_to_doc_ids"):
        for idx, doc_id in enumerate(doc_ids):
            for token_id in token_ids_batch[idx]:
                token_id = token_id.item()
                if token_id not in skip_tokens:
                    token_to_doc_ids[token_id].add(doc_id)
    return token_to_doc_ids


def build_vector_index(args, 
                     model: AutoModel, 
                     token_to_doc_ids: defaultdict[int, set[str]], 
                     documentEmbeddings: DocumentEmbeddings) -> VectorIndex:
    
    index = VectorIndex(args, model)

    if args.use_cache:
        return index.load()

    index.build(documentEmbeddings, token_to_doc_ids)
    
    index.store(args)

    return index

def build_inverted_index(args, 
                         model: AutoModel, 
                         token_to_doc_ids: defaultdict[int, set[str]], 
                         documentEmbeddings: DocumentEmbeddings, 
                         vector_index: VectorIndex) -> InvertedIndex:
    
    inverted_index_file_name = f"{args.base_path}inverted_index.pickle"

    if args.use_cache and os.path.isfile(inverted_index_file_name):
        with open(inverted_index_file_name, "rb") as f:
            return pickle.load(f)

    if os.path.isfile(inverted_index_file_name):
        os.remove(inverted_index_file_name)

    inverted_index = InvertedIndex(model, vector_index)
    for token, doc_ids in tqdm.tqdm(iterable=token_to_doc_ids.items(), desc="build_inverted_index"):
        contextualized_embs_list = []
        doc_ids_list= []
        doc_id_to_doc_emb = {}
        for doc_id in doc_ids:
            contextualized_embs, attention_mask = documentEmbeddings.get_by_token(doc_id, token)
            contextualized_embs_list.append(contextualized_embs)
            doc_ids_list.extend([doc_id] * contextualized_embs.shape[0])
            doc_id_to_doc_emb[doc_id] = mean_pooling(documentEmbeddings.get_by_document(doc_id), attention_mask).cpu().detach().numpy()

        assert len(contextualized_embs_list) != 0

        contextualized_embs = torch.cat(contextualized_embs_list, dim=0).cpu().detach().numpy()

        _, I = vector_index.search(contextualized_embs, args.index_n_neighbors)
        assert len(I) == len(contextualized_embs)
        for idx in range(len(contextualized_embs)):
            doc_id = doc_ids_list[idx]

            token_and_cluster_id_list = vector_index.map_ids_to_token_ids(I[idx])

            doc_emb = doc_id_to_doc_emb[doc_id]

            centroids = vector_index.reconstruct(I[idx])
            
            scores = np.squeeze(doc_emb @ centroids.T, 0)
            assert len(token_and_cluster_id_list) == len(scores)
            for token_and_cluster_id, score in zip(token_and_cluster_id_list, scores):
                inverted_index.add(token_and_cluster_id, (doc_id, score))

    with open(inverted_index_file_name, "wb") as f:
        pickle.dump(inverted_index, f)

    return inverted_index


def perform_searches(args, inverted_index: InvertedIndex, queries):
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



def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='scifact', help='BEIR dataset name (default scifact)')
    parser.add_argument('-l', '--dataset-length', type=int, default=-1, help='Dataset length (default -1, all dataset)')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='batch size (default 128)')
    parser.add_argument('-kmn', '--kmeans-n-clusters', type=int, default=8, help='kmeans clusters number (default 8)')
    parser.add_argument('-M', '--hnsw-M', type=int, default=32, help='the number of neighbors used in the graph. A larger M is more accurate but uses more memory (default 32)')
    parser.add_argument('-in', '--index-n-neighbors', type=int, default=8, help='index neighbors number (default 8)')
    parser.add_argument('-stk', '--search-top-k', type=int, default=1000, help='search tok k results (default 1000)')
    parser.add_argument('-sn', '--search-n-neighbors', type=int, default=3, help='search neighbors number (default 3)')
    parser.add_argument('-c', '--use-cache', action="store_true", help='use cache (default False)')
    parser.add_argument('-p', '--base-path', type=str, default='./', help='base path (default ./)')
    parser.add_argument('-pq', '--use-pq', type=bool, default=False, help='use PQ compression')
    args = parser.parse_args()
    print(f"Params: {args}")
    return args


def map_corpus(corpus, sep = " "):
    return {doc_id: (doc["title"] + sep + doc["text"]).strip() for i, (doc_id, doc) in enumerate(corpus.items())
              if args.dataset_length == -1 or i < args.dataset_length}

if __name__ == '__main__':
    # Hyperparameters
    args = get_cmd_args()
    args.dataset_length = 100
    # Data, tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID, use_fast=True)
    corpus, queries, qrels = load_dataset(dataset=args.dataset)

    corpus = map_corpus(corpus)
    print(f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}")
    dataset = CorpusDataset(corpus)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)
    
    model = load_model()
    # Indexing

    token_to_doc_ids = build_token_to_doc_ids(dataloader)
    documentEmbeddings = PQDocumentEmbeddings(model, dataset, dataloader) if args.use_pq else DocumentEmbeddings(model, dataset, dataloader)

    vector_index = build_vector_index(args, model, token_to_doc_ids, documentEmbeddings)
    inverted_index = build_inverted_index(args, model, token_to_doc_ids, documentEmbeddings, vector_index)

    # Retrieval
    results = perform_searches(args, inverted_index, queries)

    
    retriever = EvaluateRetrieval(score_function="dot")
    retrieval_result = retriever.evaluate(qrels, results, retriever.k_values)
    print(retrieval_result)
    with open("retrieval_results.txt", "a") as f:
        f.write(str(args))
        f.write("\n")
        f.write(str(retrieval_result))
        f.write("\n\n")
