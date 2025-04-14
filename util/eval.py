import logging
from typing import List

import numpy as np
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.sparse import SparseSearch
from tqdm.autonotebook import trange

#### Just some code to print debug information to stdout
#### /print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()],
                    force=True)


def eval_model(
        encode_fun,
        corpus,
        queries,
        qrels,
        k_values: List[int] = [1, 3, 5, 10, 100, 1000],
        batch_size=100,
        metric=None):
    beir_sparse_model = _build_beir_sparse_searcher(encode_fun=encode_fun, batch_size=batch_size)
    retriever = EvaluateRetrieval(beir_sparse_model, k_values=k_values, score_function="dot")
    results = retriever.retrieve(corpus, queries, query_weights=True)
    logging.info("Retriever evaluation with k in: {}".format(retriever.k_values))
    if metric:
        return retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
    else:
        return retriever.evaluate(qrels, results, retriever.k_values)


def _build_beir_sparse_searcher(encode_fun, batch_size):
    return SparseSearch(model=_SparseEncoderModel(encode_fun=encode_fun), batch_size=batch_size)


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
