from typing import Callable, Dict, Tuple
from torch.utils.data import Dataset
from common.datasets import load_dataset


class CorpusDataset(Dataset):
    def __init__(self, corpus: Dict, tokenize: Callable, lazy_loading: bool = False):
        self.doc_ids = list(corpus.keys())
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        docs = list(corpus.values())
        self.docs_len = len(docs)
        self.lazy_loading = lazy_loading
        if self.lazy_loading:
            self.docs = docs
            self.tokenize = tokenize
        else:
            self.tokenized_docs = tokenize(docs)

    def __len__(self):
        return self.docs_len

    def __getitem__(self, item):
        if self.lazy_loading:
            doc = self.docs[item]
            tokenized = self.tokenize([doc])
            return (
                self.doc_ids[item],
                tokenized["input_ids"][0],
                tokenized["attention_mask"][0],
            )
        else:
            return (
                self.doc_ids[item],
                self.tokenized_docs["input_ids"][item],
                self.tokenized_docs["attention_mask"][item],
            )

    def get_by_doc_id(self, doc_id, device="cpu"):
        idx = self.doc_id_to_idx[doc_id]
        if self.lazy_loading:
            doc = self.docs[idx]
            tokenized = self.tokenize([doc])
            return (
                tokenized["input_ids"][0].to(device).unsqueeze(0),
                tokenized["attention_mask"][0].to(device).unsqueeze(0),
            )
        else:
            return (
                self.tokenized_docs["input_ids"][idx].to(device).unsqueeze(0),
                self.tokenized_docs["attention_mask"][idx].to(device).unsqueeze(0),
            )


def get_dataset(
    tokenize: Callable,
    dataset: str | None = None,
    length: int | None = None,
    lazy_loading: bool = False,
) -> Tuple[CorpusDataset, Dict[str, str], Dict[str, Dict[str, int]]]:
    corpus, queries, qrels = load_dataset(dataset=dataset, length=length)
    sep = " "
    corpus = {
        doc_id: (doc["title"] + sep + doc["text"]).strip()
        for doc_id, doc in corpus.items()
    }
    print(
        f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}"
    )
    return CorpusDataset(corpus=corpus, tokenize=tokenize, lazy_loading=lazy_loading), queries, qrels
