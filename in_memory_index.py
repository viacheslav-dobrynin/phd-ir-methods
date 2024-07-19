from collections import defaultdict
from typing import Dict


class InMemoryInvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)

    def add(self, doc_id, doc_embedding):
      for non_zero_index in torch.nonzero(doc_embedding):
        term = non_zero_index.item()
        self.index[term].append((doc_id, doc_embedding[term].item()))

    def search(self, query_emb, top_k):
      docs = []
      for non_zero_index in torch.nonzero(query_emb):
        term = non_zero_index.item()
        docs.extend(self.index[term])
      
      return sorted(docs, key = lambda doc: doc[1], reverse = True)[:top_k]

    def avg_len(self):
      return sum(len(v) for v in self.index.values() ) / len(self.index)