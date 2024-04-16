import nltk
import pandas as pd
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import reuters, stopwords
from torch.utils.data import Dataset, DataLoader

from eval import load_dataset
from params import MAX_LENGTH, DATASET, BATCH_SIZE, EPOCHS

nltk.download('stopwords')
nltk.download('reuters')
nltk.download('wordnet')


class DocDataset(Dataset):
    def __init__(self, X, tokenizer, max_length=MAX_LENGTH):
        super().__init__()

        self.tokenizer = tokenizer
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer_simple = RegexpTokenizer(r'\w+')
        self.max_length = max_length
        self.vocabulary, self.preprocessed_docs = [], []
        self.stops = set(stopwords.words("english"))

        self.docs = X
        for doc in self.docs:
            self.vocabulary.append(self.filter_doc(doc))
            self.preprocessed_docs.append(" ".join(self.filter_doc(doc)))

        self.tokenized_docs = tokenizer(self.docs,
                                        return_tensors="pt",
                                        padding='max_length',
                                        truncation=True,
                                        max_length=max_length)

    def filter_doc(self, doc):
        words = self.tokenizer_simple.tokenize(doc)
        filtered_words = []
        for word in words:
            lemmatized = self.lemmatizer.lemmatize(word.lower())
            if lemmatized not in self.stops and not lemmatized.isdigit():
                filtered_words.append(lemmatized)
        return filtered_words

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
    dataset = DocDataset(corpus, tokenizer) # corpus
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # anneal params
    dataset_n = len(dataset)
    max_iter = len(dataloader) * EPOCHS
    print(f'{dataset_n=}')
    print(f'{max_iter=}')
    return dataloader, dataset_n, max_iter
