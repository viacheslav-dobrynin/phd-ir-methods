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

lucene.initVM()

analyzer = StandardAnalyzer()
index = "./runs/inverted_index"
delete_folder(index)
indexPath = Paths.get(index)
directory = FSDirectory.open(indexPath)
config = IndexWriterConfig(analyzer)
writer = IndexWriter(directory, config)

corpus, queries, qrels = load_dataset()
corpus = {doc_id: (doc["title"] + " " + doc["text"]).strip() for doc_id, doc in corpus.items()}
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
