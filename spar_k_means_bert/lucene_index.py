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
from org.apache.lucene.search import (
    BooleanQuery,
    BooleanClause,
    FieldExistsQuery,
    QueryRescorer,
)
from org.apache.lucene.index import ReaderUtil

import heapq, math

from common.field import to_doc_id_field
from common.path import delete_folder


class LuceneIndex:
    def __init__(self, base_path: str, use_cache: bool, threshold: torch.tensor = None):
        jcc_path = f"{base_path}tools/jcc"
        if jcc_path not in sys.path:
            sys.path.append(jcc_path)
        try:
            lucene.initVM()
        except Exception as e:
            print(f"Init error: {e}")
        self.index_path = f"{base_path}runs/spar_k_means_bert/lucene_inverted_index"
        if not use_cache:
            delete_folder(self.index_path)
        self.index_jpath = Paths.get(self.index_path)
        config = IndexWriterConfig(KeywordAnalyzer())
        self.writer = IndexWriter(FSDirectory.open(self.index_jpath), config)
        self.threshold = threshold

    def index(self, doc_id: int, token_and_cluster_id_list: list, scores: torch.tensor):
        assert len(token_and_cluster_id_list) == len(scores)
        doc = Document()
        doc.add(to_doc_id_field(doc_id))
        for token_and_cluster_id, score in zip(token_and_cluster_id_list, scores):
            if not self.threshold or score >= self.threshold:
                doc.add(
                    NumericDocValuesField(
                        token_and_cluster_id,
                        score.to(torch.float8_e4m3fn).view(dtype=torch.uint8).item(),
                    )
                )
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
                token_and_cluster_id_list = token_and_cluster_id_calculator(
                    queries[query_id]
                )
                pre = self.__build_prequery(token_and_cluster_id_list)
                candidates = searcher.search(pre, 3000)

                if len(candidates.scoreDocs) == 0:
                    continue
                leaves = searcher.getIndexReader().leaves()
                heap = []  # min-heap (score, docID)
                for sd in candidates.scoreDocs:
                    leaf_idx = ReaderUtil.subIndex(sd.doc, leaves)
                    leaf = leaves.get(leaf_idx)
                    local = sd.doc - leaf.docBase
                    r = leaf.reader()

                    total = 0.0
                    for fname in token_and_cluster_id_list:
                        dv = r.getNumericDocValues(fname)
                        if dv is None:
                            continue
                        if dv.advance(local) == local:
                            total += float(dv.longValue())

                    if total == 0.0:
                        continue
                    if len(heap) < top_k:
                        heapq.heappush(heap, (total, sd.doc))
                    else:
                        if total > heap[0][0]:
                            heapq.heapreplace(heap, (total, sd.doc))
                stored_fields = searcher.storedFields()
                query_result = {}
                while heap:
                    score, docID = heapq.heappop(heap)
                    doc = stored_fields.document(docID)
                    query_result[doc.get("doc_id")] = float(score)
                results[query_id] = query_result
        finally:
            reader.close()
        return results

    def __build_prequery(self, token_and_cluster_id_list, msm_ratio=0.1):
        b = BooleanQuery.Builder()
        for fname in token_and_cluster_id_list:
            b.add(FieldExistsQuery(fname), BooleanClause.Occur.SHOULD)
        msm = max(1, int(math.ceil(msm_ratio * max(1, len(token_and_cluster_id_list)))))
        b.setMinimumNumberShouldMatch(msm)
        return b.build()

    def __build_query(self, token_and_cluster_id_list):
        field_sources = []
        for token_and_cluster_id in token_and_cluster_id_list:
            field_sources.append(FloatFieldSource(token_and_cluster_id))
        return FunctionQuery(SumFloatFunction(field_sources))
