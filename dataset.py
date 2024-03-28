import nltk
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import reuters, stopwords
from torch.utils.data import Dataset

from params import MAX_LENGTH

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
