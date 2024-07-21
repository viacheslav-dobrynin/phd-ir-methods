from collections import defaultdict
from typing import Dict
import torch

class InMemoryInvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)

    def add(self, doc_id, doc_embedding):
      for non_zero_index in torch.nonzero(doc_embedding):
        term = non_zero_index.item()
        self.index[term].append((doc_id, doc_embedding[term].item()))

    def search(self, query_emb, top_k):
      score_by_doc_id_map = defaultdict(float)
      
      for non_zero_index in torch.nonzero(query_emb):
        term = non_zero_index.item()
        for doc_id, value in self.index[term]:
          score_by_doc_id_map[doc_id] += query_emb[non_zero_index].item() * value
      
      return sorted(score_by_doc_id_map.items(), key=lambda e: e[1], reverse=True)[:top_k]

    def avg_len(self):
      return sum(len(v) for v in self.index.values() ) / len(self.index)