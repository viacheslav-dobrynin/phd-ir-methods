import os
import shutil

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, FloatDocValuesField, StringField, Field
from org.apache.lucene.index import DirectoryReader, IndexWriterConfig, IndexWriter
from org.apache.lucene.search import BooleanClause, BooleanQuery, BoostQuery, IndexSearcher
from org.apache.lucene.store import FSDirectory
from ru.itmo.sparsifiermodel.query import FieldValueAsScoreQuery

from dataset import load_dataset


def delete_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"The folder '{path}' and its contents have been successfully deleted.")
    else:
        print(f"The folder '{path}' does not exist.")


def to_field_name(i):
    return f"term_{i}"


def build_query(query):
    builder = BooleanQuery.Builder()
    for i, value in enumerate(query):
        if value != 0.0:
            builder.add(BoostQuery(FieldValueAsScoreQuery(to_field_name(i)), value), BooleanClause.Occur.SHOULD)
    return builder.build()


lucene.initVM()

analyzer = StandardAnalyzer()
index = "./runs/inverted_index"
delete_folder(index)
indexPath = Paths.get(index)
directory = FSDirectory.open(indexPath)
config = IndexWriterConfig(analyzer)
writer = IndexWriter(directory, config)

corpus, queries, qrels = load_dataset()
doc_ids = list(corpus.keys())
query_ids = list(queries.keys())
documents = [corpus[doc_id] for doc_id in doc_ids]
sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in documents]

for doc_id, emb in enumerate([[0.0, 3.5, 0.0], [4.5, 0.0, 5.5], [0.0, 6.5, 0.0]]):
    doc = Document()
    doc.add(StringField("doc_id", str(doc_id), Field.Store.YES))
    for i, value in enumerate(emb):
        if value != 0.0:
            doc.add(FloatDocValuesField(to_field_name(i), value))
    writer.addDocument(doc)
writer.close()

reader = DirectoryReader.open(FSDirectory.open(indexPath))
searcher = IndexSearcher(reader)
query = build_query([1.0, 2.0, 0.0])

hits = searcher.search(query, 10).scoreDocs
storedFields = searcher.storedFields()
for hit in hits:
    hitDoc = storedFields.document(hit.doc)
    print(f"{hitDoc=}, {hit.score=}, {hitDoc['doc_id']=}")

reader.close()
directory.close()
