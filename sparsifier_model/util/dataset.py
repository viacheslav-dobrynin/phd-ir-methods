import nltk
import pandas as pd
from nltk.corpus import reuters
from torch.utils.data import Dataset, DataLoader

from common.datasets import load_dataset
from sparsifier_model.config import Config


class DocDataset(Dataset):
    def __init__(self, config: Config, docs, tokenizer):
        super().__init__()
        self.docs_len = len(docs)
        self.tokenized_docs = tokenizer(docs,
                                        return_tensors="pt",
                                        padding='max_length',
                                        truncation=True,
                                        max_length=config.max_length)

    def __len__(self):
        return self.docs_len

    def __getitem__(self, item):
        return self.tokenized_docs['input_ids'][item], self.tokenized_docs['attention_mask'][item]


def get_reuters_raw(num_doc=100):
    nltk.download('stopwords')
    nltk.download('reuters')
    nltk.download('wordnet')
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


def get_dataloader(config: Config, tokenizer):
    # corpus = get_corpus(config.dataset)
    # Create dataset and dataloader
    # dataset = DocDataset(config, corpus, tokenizer)
    # dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # del corpus
    from datasets import load_dataset
    corpus = load_dataset("nlphuji/flickr30k")['test']
    def collate_pil(batch):
        # batch — список словарей; поле с изображением называется 'image' (PIL.Image)
        return [record['image'] for record in batch]
    dataloader = DataLoader(corpus, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=collate_pil)
    # anneal params
    dataset_n = len(corpus)
    max_iter = len(dataloader) * config.epochs
    print(f'{dataset_n=}')
    print(f'{max_iter=}')
    return dataloader, dataset_n, max_iter
