import itertools

import lucene
import torch
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, FloatDocValuesField, StringField, Field
from org.apache.lucene.index import DirectoryReader, IndexWriterConfig, IndexWriter
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import FSDirectory
from tqdm.autonotebook import trange

from dataset import load_dataset
from util.path import delete_folder
from util.field import to_field_name, to_doc_id_field
from util.search import build_query


class Runner:
    def __init__(self, encode_fun):
        try:
            lucene.initVM()
        except ValueError as e:
            print(f'Init error: {e}')
        self.encode = encode_fun
        self.analyzer = StandardAnalyzer()
        self.index_path = "./runs/inverted_index"
        self.index_jpath = Paths.get(self.index_path)
        corpus, self.queries, self.qrels = load_dataset()
        self.corpus = {doc_id: (doc["title"] + " " + doc["text"]).strip() for doc_id, doc in corpus.items()}

    def index(self, batch_size=300):
        config = IndexWriterConfig(self.analyzer)
        writer = IndexWriter(FSDirectory.open(self.index_jpath), config)

        try:
            corpus_items = self.corpus.items()
            for start_idx in trange(0, len(self.corpus), batch_size, desc="docs"):
                batch = tuple(itertools.islice(corpus_items, start_idx, start_idx + batch_size))
                doc_ids, docs = list(zip(*batch))
                emb_batch = self.encode(docs)
                doc, prev_batch_idx = Document(), None
                for batch_idx, term in torch.nonzero(emb_batch):
                    if prev_batch_idx is not None and prev_batch_idx != batch_idx:
                        doc.add(to_doc_id_field(doc_ids[prev_batch_idx]))
                        writer.addDocument(doc)
                        doc = Document()
                    doc.add(FloatDocValuesField(to_field_name(term.item()), emb_batch[batch_idx, term].item()))
                    prev_batch_idx = batch_idx
                doc.add(to_doc_id_field(doc_ids[prev_batch_idx]))
                writer.addDocument(doc)
        finally:
            writer.close()

    def search(self, top_k=10):
        reader = DirectoryReader.open(FSDirectory.open(self.index_jpath))
        searcher = IndexSearcher(reader)
        results = {}

        try:
            query_ids = list(self.queries.keys())
            for query_id in query_ids:
                query_emb = self.encode([self.queries[query_id]])[0]
                hits = searcher.search(build_query(query_emb), top_k).scoreDocs
                stored_fields = searcher.storedFields()
                query_result = {}
                for hit in hits:
                    hit_doc = stored_fields.document(hit.doc)
                    query_result[hit_doc["doc_id"]] = hit.score
                results[query_id] = query_result
        finally:
            reader.close()
        return results

    def delete_index(self):
        delete_folder(self.index_path)


if __name__ == '__main__':
    runner = Runner(encode_fun=lambda docs: torch.tensor([[12.0, .0, 15.0], [.0, 4.0, .0], [20.0, 30.5, .0]]))
    runner.delete_index()
    runner.index()
    search_results = runner.search()
    print(f"{search_results=}")
