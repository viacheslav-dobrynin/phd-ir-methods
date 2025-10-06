from typing import Callable, Dict, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from common.datasets import load_dataset


class CorpusDataset(Dataset):
    def __init__(self, corpus: Dict, tokenize: Callable, lazy_loading: bool):
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


def get_dataset(tokenize: Callable, args) -> Tuple[CorpusDataset, Dict[str, str], Dict[str, Dict[str, int]]]:
    corpus, queries, qrels = load_dataset(dataset=args.dataset, length=args.dataset_length)
    sep = " "
    corpus = {
        doc_id: (doc["title"] + sep + doc["text"]).strip()
        for doc_id, doc in corpus.items()
    }
    print(
        f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}"
    )
    return CorpusDataset(corpus=corpus, tokenize=tokenize, lazy_loading=args.lazy_loading), queries, qrels


def get_dataloader(dataset: CorpusDataset, args, pad_token_id: int) -> DataLoader:
    def _collate_fn(batch):
        doc_ids, input_ids_list, attention_mask_list = zip(*batch)
        # Pad sequences to the same length
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
        attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        return list(doc_ids), input_ids_padded, attention_mask_padded
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=_collate_fn if args.lazy_loading else None
    )
