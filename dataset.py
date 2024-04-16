import os

import nltk
import pandas as pd
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from nltk.corpus import reuters
from torch.utils.data import Dataset, DataLoader

from params import MAX_LENGTH, DATASET, BATCH_SIZE, EPOCHS

nltk.download('stopwords')
nltk.download('reuters')
nltk.download('wordnet')


class DocDataset(Dataset):
    def __init__(self, X, tokenizer, max_length=MAX_LENGTH):
        super().__init__()
        self.docs = X
        self.tokenized_docs = tokenizer(self.docs,
                                        return_tensors="pt",
                                        padding='max_length',
                                        truncation=True,
                                        max_length=max_length)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, item):
        return self.tokenized_docs['input_ids'][item], self.tokenized_docs['attention_mask'][item]


def get_reuters_raw(num_doc=100):
    file_list = reuters.fileids()
    if num_doc:
        corpus = [reuters.raw(file_list[i]) for i in range(num_doc)]
    else:
        corpus = [reuters.raw(file_list[i]) for i in range(len(file_list))]
    return corpus


def load_dataset(dataset="scifact"):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    print("Dataset downloaded here: {}".format(data_path))
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")  # or split = "train" or "dev"
    return corpus, queries, qrels


def get_corpus():
    if DATASET == "reuters":
        corpus = get_reuters_raw(num_doc=None)
        corpus_df = pd.DataFrame(corpus, columns=["Text"])
        corpus = corpus_df["Text"].drop_duplicates().values.tolist()
    elif "msmarco" in DATASET:
        corpus_all, _, _ = load_dataset(dataset="msmarco")
        count = int(DATASET.split('_')[1])
        corpus_gen = (value for i, value in enumerate(corpus_all.values()) if i < count)
        sep = " "
        corpus = [(doc["title"] + sep + doc["text"]).strip() for doc in corpus_gen]
        del corpus_gen
        del corpus_all
    else:
        raise ValueError("Unknown dataset:", DATASET)

    return corpus


def get_dataloader(tokenizer):
    corpus = get_corpus()
    # Create dataset and dataloader
    dataset = DocDataset(corpus, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # anneal params
    dataset_n = len(dataset)
    max_iter = len(dataloader) * EPOCHS
    print(f'{dataset_n=}')
    print(f'{max_iter=}')
    return dataloader, dataset_n, max_iter
