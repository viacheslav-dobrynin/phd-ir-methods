import sys
import tools.lucene as lucene
import torch
import tqdm
from java.nio.file import Paths
from org.apache.lucene.analysis.core import KeywordAnalyzer
from org.apache.lucene.document import Document
from org.apache.lucene.document import FeatureField
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import IndexWriter
from org.apache.lucene.index import IndexWriterConfig
from org.apache.lucene.search import BooleanClause, BooleanQuery
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import FSDirectory

import math
from common.field import to_doc_id_field
from common.path import delete_folder


class LuceneInvertedIndex:
    def __init__(self, index_path: str, threshold: float | None = None) -> None:
        jcc_path = "./tools/jcc"
        if jcc_path not in sys.path:
            sys.path.append(jcc_path)
        try:
            lucene.initVM()
        except Exception as e:
            print(f"Init error: {e}")
        self.index_path = index_path
        self.index_jpath = Paths.get(self.index_path)
        config = IndexWriterConfig(KeywordAnalyzer())
        self.writer = IndexWriter(FSDirectory.open(self.index_jpath), config)
        self.field_name = "sparse_vector"
        self.threshold = threshold

    def index(self, doc_id: int, terms: list[str], scores: torch.tensor):
        assert len(terms) == len(scores)
        doc = Document()
        doc.add(to_doc_id_field(doc_id))
        for term, score in zip(terms, scores):
            if self.threshold is None or score >= self.threshold:
                doc.add(FeatureField(self.field_name, term, score.item()))
        self.writer.addDocument(doc)

    def complete_indexing(self, merge_to_one_segment: bool = True):
        if merge_to_one_segment:
            self.writer.forceMerge(1, True)
        self.writer.commit()
        self.writer.close()

    def delete_index(self):
        delete_folder(self.index_path)

    def search_by_query(
        self,
        searcher,
        query_terms: list[str],
        query_scores: list | None = None,
        top_k=1000,
        msm_ratio: float = 0.0,
    ):
        if query_scores is None:
            query_scores = [1] * len(query_terms)
        else:
            assert len(query_terms) == len(query_scores)
        max_w = max(query_scores)
        if max_w > 64.0:
            scale = 64.0 / max_w
            for i in range(len(query_terms)):
                query_scores[i] *= scale
        b = BooleanQuery.Builder()
        if msm_ratio > 0:
            msm = max(1, int(math.ceil(msm_ratio * len(query_terms))))
            b.setMinimumNumberShouldMatch(msm)

        for term, score in zip(query_terms, query_scores):
            b.add(
                FeatureField.newLinearQuery(self.field_name, term, float(score)),
                BooleanClause.Occur.SHOULD,
            )
        query = b.build()
        hits = searcher.search(query, top_k).scoreDocs
        stored_fields = searcher.storedFields()
        results = {stored_fields.document(hit.doc)["doc_id"]: hit.score for hit in hits}
        return results

    def get_reader_and_searcher(self):
        reader = DirectoryReader.open(FSDirectory.open(self.index_jpath))
        searcher = IndexSearcher(reader)
        return reader, searcher

    def size(self):
        try:
            reader, _ = self.get_reader_and_searcher()
            num_docs = reader.numDocs()
            reader.close()
            return num_docs
        except:
            return 0

    def search(
        self,
        queries,
        sparse_vector_calculator,
        top_k=1000,
        msm_ratio: float = 0.0,
    ):
        reader, searcher = self.get_reader_and_searcher()
        results = {}
        try:
            for query_id, query in tqdm.tqdm(queries.items(), desc="search"):
                query_terms, query_scores = sparse_vector_calculator(query)
                results[query_id] = self.search_by_query(
                    searcher=searcher,
                    query_terms=query_terms,
                    query_scores=query_scores,
                    top_k=top_k,
                    msm_ratio=msm_ratio,
                )
        finally:
            reader.close()
        return results
