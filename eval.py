import os

import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.sparse import SparseSearch
from tqdm.autonotebook import trange


def load_dataset(dataset="scifact"):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    print("Dataset downloaded here: {}".format(data_path))
    data_path = f"datasets/{dataset}"
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")  # or split = "train" or "dev"
    return corpus, queries, qrels


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
