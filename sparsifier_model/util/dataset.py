import nltk
import pandas as pd
from nltk.corpus import reuters
from torch.utils.data import Dataset, DataLoader

from sparsifier_model.params import MAX_LENGTH, EPOCHS
from util.datasets import load_dataset

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


def get_dataloader(tokenizer, dataset_name: str, batch_size: int):
    corpus = get_corpus(dataset_name)
    # Create dataset and dataloader
    dataset = DocDataset(corpus, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    del corpus
    # anneal params
    dataset_n = len(dataset)
    max_iter = len(dataloader) * EPOCHS
    print(f'{dataset_n=}')
    print(f'{max_iter=}')
    return dataloader, dataset_n, max_iter
