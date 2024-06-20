from collections import defaultdict
from typing import Dict

import torch
from tqdm.autonotebook import trange


class InvertedIndexOnDict:
    def __init__(self, encode_fun):
        self.index = defaultdict(list)
        self.encode = encode_fun

    def write(self, embs):
        if len(self.index) != 0:
            raise RuntimeError("Already indexed")
        nonzero_embs = torch.nonzero(embs)
        doc_ids = nonzero_embs[:, 0]
        terms = nonzero_embs[:, 1]
        for idx, term in enumerate(terms):
            doc_id = doc_ids[idx]
            self.index[term.item()].append((doc_id.item(), embs[doc_id, term].item()))

    def search(self, queries: Dict[str, str], top_k: int):
        results = {}
        query_ids = list(queries.keys())
        for q_idx in trange(0, len(queries), desc='query'):
            q_id = query_ids[q_idx]
            query_terms = self.encode(queries[q_id])
            score_by_doc_id_map = defaultdict(float)
            for query_term_idx in torch.nonzero(query_terms):
                for doc_id, value in self.index[query_term_idx.item()]:
                    score_by_doc_id_map[doc_id] += query_terms[query_term_idx].item() * value
            results[q_id] = dict(sorted(score_by_doc_id_map.items(), key=lambda e: e[1], reverse=True)[:top_k])
        return results
