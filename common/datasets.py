import os

from beir import util
from beir.datasets.data_loader import GenericDataLoader


def load_dataset(dataset: str = None, split: str = "test", length: int = None):
    if not dataset:
        dataset = "scifact"
    if length:
        assert length >= 1
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    print("Dataset downloaded here: {}".format(data_path))
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)  # or split = "train" or "dev"
    if not length:
        return corpus, queries, qrels
    filtered_corpus = {doc_id: doc for i, (doc_id, doc) in enumerate(corpus.items()) if i < length}
    filtered_qrels = {
        query_id: {
            doc_id: rel
            for doc_id, rel in doc_id_to_rel.items()
            if doc_id in filtered_corpus.keys()
        }
        for query_id, doc_id_to_rel in qrels.items()
        if any(doc_id in filtered_corpus.keys() for doc_id in doc_id_to_rel.keys())
    }
    filtered_queries = {query_id: queries[query_id]
                        for query_id in filtered_qrels}
    del corpus
    del queries
    del qrels
    return filtered_corpus, filtered_queries, filtered_qrels
