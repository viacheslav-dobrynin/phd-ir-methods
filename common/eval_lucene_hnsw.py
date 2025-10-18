import argparse
import sys
import time

import torch
import tqdm
from transformers import AutoTokenizer
from common.bench import run_bench, calc_stats
from common.datasets import load_dataset
from common.encode_dense_fun_builder import build_encode_dense_fun
from common.model import load_model
from common.path import delete_folder
from spar_k_means_bert.util.eval import eval_with_dot_score_function
import tools.lucene as lucene

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search import IndexSearcher, KnnFloatVectorQuery
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import (
    VectorSimilarityFunction,
    IndexWriterConfig,
    IndexWriter,
    DirectoryReader,
)
from java.nio.file import Paths
from org.apache.lucene.document import Document, KnnFloatVectorField, StringField, Field

parser = argparse.ArgumentParser()
parser.add_argument('--eval-or-bench', type=str, default='eval', help='eval or bench (default eval)')
parser.add_argument('-l', '--dataset-length', type=int, default=None, help='Dataset length (default None, all dataset)')
args = parser.parse_args()
print(f"Params: {args}")

backbone_model_id = "sentence-transformers/msmarco-distilbert-dot-v5"
tokenizer = AutoTokenizer.from_pretrained(backbone_model_id, use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(backbone_model_id, device)
encode_dense = build_encode_dense_fun(tokenizer=tokenizer, model=model, device=device)

jcc_path = "./tools/jcc"
if jcc_path not in sys.path:
    sys.path.append(jcc_path)
try:
    lucene.initVM()
except Exception as e:
    print(f"Init error: {e}")

analyzer = StandardAnalyzer()
index_path = "./runs/common/lucene_hnsw_index"
index_jpath = Paths.get(index_path)

corpus, queries, qrels = load_dataset(dataset="msmarco", split="dev", length=args.dataset_length)
print(
    f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}"
)

try:
    reader = DirectoryReader.open(FSDirectory.open(index_jpath))
    num_docs = reader.numDocs()
    reader.close()
except:
    num_docs = 0

if num_docs == 0:
    delete_folder(index_path)
    config = IndexWriterConfig(analyzer)
    writer = IndexWriter(FSDirectory.open(index_jpath), config)
    sep = " "
    try:
        for doc_id, doc in tqdm.tqdm(iterable=corpus.items(), desc="build_hnsw"):
            doc = (doc["title"] + sep + doc["text"]).strip()
            doc_emb = encode_dense(doc).cpu()[0].tolist()
            lucene_document = Document()
            lucene_document.add(StringField("doc_id", doc_id, Field.Store.YES))
            lucene_document.add(
                KnnFloatVectorField("vector", doc_emb, VectorSimilarityFunction.DOT_PRODUCT)
            )
            writer.addDocument(lucene_document)
    finally:
        writer.forceMerge(1, True)
        writer.commit()
        writer.close()
reader = DirectoryReader.open(FSDirectory.open(index_jpath))
searcher = IndexSearcher(reader)
num_docs = reader.numDocs()
print("HNSW index size:", num_docs)

def search(query, top_k):
    query_emb = encode_dense(query).cpu()[0].tolist()
    query = KnnFloatVectorQuery("vector", query_emb, top_k)
    hits = searcher.search(query, top_k).scoreDocs
    stored_fields = searcher.storedFields()
    query_result = {}
    for hit in hits:
        hit_doc = stored_fields.document(hit.doc)
        query_result[hit_doc["doc_id"]] = hit.score
    return query_result

top_k = 1000
if args.eval_or_bench == "eval":
    start = time.time()
    results = {}
    try:
        for query_id, query in tqdm.tqdm(iterable=queries.items(), desc="search"):
            results[query_id] = search(query=query, top_k=top_k)
    finally:
        reader.close()
    print("Search time:", time.time() - start)

    ndcg, _map, recall, precision, mrr = eval_with_dot_score_function(qrels, results)
    print(ndcg)
    print(mrr)
    print(_map)
    print(recall)
    print(precision)
else:
    try:
        # Warmup
        run_bench(func_to_bench=lambda: search("warmup benchmark query for measuring search latency", top_k), warmup=1000, repeats=10)
        # Bench
        queries = list(queries.values())
        repeats = max(2000, len(queries)) // len(queries)
        samples = []
        for query in queries:
            query_samples = run_bench(func_to_bench=lambda: search(query, top_k), warmup=0, repeats=repeats)
            samples.extend(query_samples)
        calc_stats("Bench results", samples)
    finally:
        reader.close()
