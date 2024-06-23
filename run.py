import itertools

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, FloatDocValuesField, StringField, Field
from org.apache.lucene.index import DirectoryReader, IndexWriterConfig, IndexWriter
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import FSDirectory
from tqdm.autonotebook import trange

from dataset import load_dataset
from util.path import delete_folder
from util.field import to_field_name
from util.search import build_query


class Runner:
    def __init__(self):
        lucene.initVM()
        self.analyzer = StandardAnalyzer()
        self.index_path = "./runs/inverted_index"
        self.index_jpath = Paths.get(self.index_path)
        self.corpus, self.queries, self.qrels = load_dataset()

    def index(self):
        delete_folder(self.index_path)
        config = IndexWriterConfig(self.analyzer)
        writer = IndexWriter(FSDirectory.open(self.index_jpath), config)

        corpus = {doc_id: (doc["title"] + " " + doc["text"]).strip() for doc_id, doc in self.corpus.items()}
        batch_size = 100
        encode = None
        corpus_items = corpus.items()
        for start_idx in trange(0, len(corpus), batch_size, desc="docs"):
            batch = tuple(itertools.islice(corpus_items, start_idx, start_idx + batch_size))
            doc_ids, docs = list(zip(*batch))

        for doc_id, emb in enumerate([[0.0, 3.5, 0.0], [4.5, 0.0, 5.5], [0.0, 6.5, 0.0]]):
            doc = Document()
            doc.add(StringField("doc_id", str(doc_id), Field.Store.YES))
            for i, value in enumerate(emb):
                if value != 0.0:
                    doc.add(FloatDocValuesField(to_field_name(i), value))
            writer.addDocument(doc)
        writer.close()

    def search(self):
        reader = DirectoryReader.open(FSDirectory.open(self.index_jpath))
        searcher = IndexSearcher(reader)
        query = build_query([1.0, 2.0, 0.0])

        hits = searcher.search(query, 10).scoreDocs
        storedFields = searcher.storedFields()
        for hit in hits:
            hitDoc = storedFields.document(hit.doc)
            print(f"{hitDoc=}, {hit.score=}, {hitDoc['doc_id']=}")

        reader.close()


if __name__ == '__main__':
    runner = Runner()
    runner.index()
    runner.search()
