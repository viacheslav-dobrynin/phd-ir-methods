import os
import pickle
from collections import defaultdict

import torch
import tqdm


class InMemoryInvertedIndex:
    def __init__(self, base_path: str, use_cache: bool):
        self.index_file = f"{base_path}inverted_index.pickle"
        self.inverted_index = defaultdict(set)
        if os.path.isfile(self.index_file):
            if use_cache:
                with open(self.index_file, "rb") as f:
                    self.inverted_index = pickle.load(f)
            else:
                os.remove(self.index_file)

    def index(self, doc_id: int, token_and_cluster_id_list: list, scores: torch.tensor):
        assert len(token_and_cluster_id_list) == len(scores)
        scores = scores.tolist()
        for token_and_cluster_id, score in zip(token_and_cluster_id_list, scores):
            self.inverted_index[token_and_cluster_id].add((doc_id, score))

    def complete_indexing(self):
        with open(self.index_file, "wb") as f:
            pickle.dump(self.inverted_index, f)

    def size(self):
        return len(self.inverted_index)

    def search(self, queries, token_and_cluster_id_calculator, top_k=1000):
        results = {}
        query_ids = list(queries.keys())
        for query_id in tqdm.tqdm(iterable=query_ids, desc="search"):
            token_and_cluster_id_list = token_and_cluster_id_calculator(queries[query_id])
            doc_id_to_score = defaultdict(float)
            for token_and_cluster_id in token_and_cluster_id_list:
                for doc_id, score in self.inverted_index[token_and_cluster_id]:
                    doc_id_to_score[doc_id] += score
            doc_id_and_score_list = sorted(doc_id_to_score.items(), key=lambda e: e[1], reverse=True)[:top_k]
            results[query_id] = dict(doc_id_and_score_list)
        return results
