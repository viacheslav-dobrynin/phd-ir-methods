import os

import nltk
import pandas as pd
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from nltk.corpus import reuters
from torch.utils.data import Dataset, DataLoader

from params import MAX_LENGTH, BATCH_SIZE, EPOCHS

nltk.download('stopwords')
nltk.download('reuters')
nltk.download('wordnet')


class DocDataset(Dataset):
    def __init__(self, docs, tokenizer, max_length=MAX_LENGTH):
        super().__init__()
        self.docs_len = len(docs)
        self.tokenized_docs = tokenizer(docs,
                                        return_tensors="pt",
                                        padding='max_length',
                                        truncation=True,
                                        max_length=max_length)

    def __len__(self):
        return self.docs_len

    def __getitem__(self, item):
        return self.tokenized_docs['input_ids'][item], self.tokenized_docs['attention_mask'][item]


def get_reuters_raw(num_doc=100):
    file_list = reuters.fileids()
    if num_doc:
        corpus = [reuters.raw(file_list[i]) for i in range(num_doc)]
    else:
        corpus = [reuters.raw(file_list[i]) for i in range(len(file_list))]
    return corpus


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


def get_corpus(dataset_name: str):
    if dataset_name == "reuters":
        corpus = get_reuters_raw(num_doc=None)
        corpus_df = pd.DataFrame(corpus, columns=["Text"])
        corpus = corpus_df["Text"].drop_duplicates().values.tolist()
    elif "msmarco" in dataset_name:
        count = int(dataset_name.split('_')[1])
        corpus, _, _ = load_dataset(dataset="msmarco", length=count)
        sep = " "
        corpus = [(doc["title"] + sep + doc["text"]).strip() for doc in corpus.values()]
    else:
        raise ValueError("Unknown dataset:", dataset_name)

    return corpus


def get_dataloader(tokenizer, dataset_name: str):
    corpus = get_corpus(dataset_name)
    # Create dataset and dataloader
    dataset = DocDataset(corpus, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    del corpus
    # anneal params
    dataset_n = len(dataset)
    max_iter = len(dataloader) * EPOCHS
    print(f'{dataset_n=}')
    print(f'{max_iter=}')
    return dataloader, dataset_n, max_iter
