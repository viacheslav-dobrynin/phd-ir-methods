import numpy as np
from tqdm.autonotebook import trange
from beir.retrieval.search.sparse import SparseSearch


def build_beir_sparse_searcher(encode_fun):
    return SparseSearch(model=_SparseEncoderModel(encode_fun=encode_fun))  # batch_size=10


class _SparseEncoderModel:
    def __init__(self, encode_fun):
        self.encode = encode_fun
        self.sep = " "

    def encode_query(self, query: str, **kwargs):
        return self.encode([query])[0].cpu().detach().numpy()

    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() for doc in corpus]
        emb_batches = []
        for start_idx in trange(0, len(sentences), batch_size, desc="docs"):
            emb_batch = self.encode(sentences[start_idx: start_idx + batch_size])
            emb_batches.append(emb_batch.cpu().detach().numpy())
        embs = np.concatenate(emb_batches)
        return embs
