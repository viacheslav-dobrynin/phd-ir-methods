from typing import Callable, Dict, Tuple
from torch.utils.data import Dataset
from common.datasets import load_dataset


class CorpusDataset(Dataset):
    def __init__(self, corpus: Dict, tokenize: Callable):
        self.doc_ids = list(corpus.keys())
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        docs = list(corpus.values())
        self.docs_len = len(docs)
        self.tokenized_docs = tokenize(docs)

    def __len__(self):
        return self.docs_len

    def __getitem__(self, item):
        return (
            self.doc_ids[item],
            self.tokenized_docs["input_ids"][item],
            self.tokenized_docs["attention_mask"][item],
        )

    def get_by_doc_id(self, doc_id, device="cpu"):
        idx = self.doc_id_to_idx[doc_id]
        return (
            self.tokenized_docs["input_ids"][idx].to(device).unsqueeze(0),
            self.tokenized_docs["attention_mask"][idx].to(device).unsqueeze(0),
        )


def get_dataset(
    tokenize: Callable,
    dataset: str | None = None,
    length: int | None = None,
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
    return CorpusDataset(corpus=corpus, tokenize=tokenize), queries, qrels
