from torch.utils.data import Dataset


class CorpusDataset(Dataset):
    def __init__(self, corpus, tokenize):
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
