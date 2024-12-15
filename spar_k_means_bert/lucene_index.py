import sys
import torch
import tqdm
import tools.lucene as lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.core import KeywordAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.index import IndexWriter
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.document import Document
from org.apache.lucene.document import NumericDocValuesField
from org.apache.lucene.queries.function import FunctionQuery
from org.apache.lucene.queries.function.valuesource import FloatFieldSource
from org.apache.lucene.queries.function.valuesource import SumFloatFunction

from util.field import to_doc_id_field
from util.path import delete_folder


class LuceneIndex:
    def __init__(self, base_path: str, use_cache: bool):
        jcc_path = '/home/slava/IdeaProjects/sparsifier-model/tools/jcc'  # TODO: use var for this
        if jcc_path not in sys.path:
            sys.path.append(jcc_path)
        try:
            lucene.initVM()
        except Exception as e:
            print(f'Init error: {e}')
        self.index_path = f"{base_path}runs/inverted_index"
        if not use_cache:
            delete_folder(self.index_path)
        self.index_jpath = Paths.get(self.index_path)
        config = IndexWriterConfig(KeywordAnalyzer())
        self.writer = IndexWriter(FSDirectory.open(self.index_jpath), config)

    def index(self, doc_id: int, token_and_cluster_id_list: list, scores: torch.tensor):
        assert len(token_and_cluster_id_list) == len(scores)
        doc = Document()
        doc.add(to_doc_id_field(doc_id))
        scores = scores.to(torch.float8_e4m3fn).view(dtype=torch.uint8).tolist()
        for token_and_cluster_id, score in zip(token_and_cluster_id_list, scores):
            doc.add(NumericDocValuesField(token_and_cluster_id, score))
        self.writer.addDocument(doc)

    def complete_indexing(self):
        self.writer.forceMerge(1, True)
        self.writer.commit()

    def size(self):
        try:
            reader = DirectoryReader.open(FSDirectory.open(self.index_jpath))
            num_docs = reader.numDocs()
            reader.close()
            return num_docs
        except:
            return 0

    def delete_index(self):
        delete_folder(self.index_path)

    def search(self, queries, token_and_cluster_id_calculator, top_k=1000):
        reader = DirectoryReader.open(FSDirectory.open(self.index_jpath))
        searcher = IndexSearcher(reader)
        results = {}

        try:
            query_ids = list(queries.keys())
            for query_id in tqdm.tqdm(iterable=query_ids, desc="search"):
                query = self.__build_query(token_and_cluster_id_calculator(queries[query_id]))
                hits = searcher.search(query, top_k).scoreDocs
                stored_fields = searcher.storedFields()
                query_result = {}
                for hit in hits:
                    hit_doc = stored_fields.document(hit.doc)
                    query_result[hit_doc["doc_id"]] = hit.score
                results[query_id] = query_result
        finally:
            reader.close()
        return results

    def __build_query(self, token_and_cluster_id_list):
        field_sources = []
        for token_and_cluster_id in token_and_cluster_id_list:
            field_sources.append(FloatFieldSource(token_and_cluster_id))
        return FunctionQuery(SumFloatFunction(field_sources))


if __name__ == '__main__':
    index = LuceneIndex()
    index.complete_indexing()
    print(KeywordAnalyzer())
