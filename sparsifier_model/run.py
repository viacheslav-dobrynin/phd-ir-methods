import time
import itertools
import sys
from transformers import AutoTokenizer
import wandb

from spar_k_means_bert.util.eval import eval_with_dot_score_function
import tools.lucene as lucene
import torch
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, FloatDocValuesField, StringField, Field
from org.apache.lucene.index import DirectoryReader, IndexWriterConfig, IndexWriter
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import FSDirectory
from tqdm.autonotebook import trange

from common.datasets import load_dataset
from common.path import delete_folder
from common.field import to_field_name, to_doc_id_field
from common.search import build_query
from common.in_memory_index import InMemoryInvertedIndex
from sparsifier_model.config import Config, ModelType
from sparsifier_model.k_sparse.model import Autoencoder
from sparsifier_model.util.model import build_encode_sparse_fun


class InMemoryIndexRunner:
    def __init__(self, encode_fun, dataset=None, docs_number=None):
        self.encode = encode_fun
        self.index_path = "./runs/sparsifier_model/in_memoty_inverted_index"
        corpus, self.queries, self.qrels = load_dataset(
            dataset=dataset, length=docs_number
        )
        self.corpus = {
            doc_id: (doc["title"] + " " + doc["text"]).strip()
            for doc_id, doc in corpus.items()
        }
        self.inverted_index = InMemoryInvertedIndex()

    def index(self, batch_size=300):
        corpus_items = self.corpus.items()
        for start_idx in trange(0, len(self.corpus), batch_size, desc="docs"):
            batch = tuple(
                itertools.islice(corpus_items, start_idx, start_idx + batch_size)
            )
            doc_ids, docs = list(zip(*batch))
            emb_batch = self.encode(docs)
            for i in range(len(emb_batch)):
                self.inverted_index.add(doc_ids[i], emb_batch[i])

    def search(self, top_k=10):
        results = {}

        query_ids = list(self.queries.keys())
        for query_id in query_ids:
            query_emb = self.encode([self.queries[query_id]])[0]
            hits = self.inverted_index.search(query_emb, top_k)

            query_result = {}
            for hit in hits:
                query_result[hit[0]] = hit[1]
            results[query_id] = query_result
        return results


class LuceneRunner:
    def __init__(self, encode_fun, dataset=None, docs_number=None):
        jcc_path = f"./tools/jcc"
        if jcc_path not in sys.path:
            sys.path.append(jcc_path)
        try:
            lucene.initVM()
        except Exception as e:
            print(f"Init error: {e}")
        self.encode = encode_fun
        self.analyzer = StandardAnalyzer()
        self.index_path = "./runs/sparsifier_model/lucene_inverted_index"
        self.index_jpath = Paths.get(self.index_path)
        corpus, self.queries, self.qrels = load_dataset(
            dataset=dataset, length=docs_number
        )
        self.corpus = {
            doc_id: (doc["title"] + " " + doc["text"]).strip()
            for doc_id, doc in corpus.items()
        }

    def index(self, batch_size=300):
        config = IndexWriterConfig(self.analyzer)
        writer = IndexWriter(FSDirectory.open(self.index_jpath), config)

        try:
            corpus_items = self.corpus.items()
            for start_idx in trange(0, len(self.corpus), batch_size, desc="docs"):
                batch = tuple(
                    itertools.islice(corpus_items, start_idx, start_idx + batch_size)
                )
                doc_ids, docs = list(zip(*batch))
                emb_batch = self.encode(docs)
                doc, prev_batch_idx = Document(), None
                for batch_idx, term in torch.nonzero(emb_batch):
                    if prev_batch_idx is not None and prev_batch_idx != batch_idx:
                        doc.add(to_doc_id_field(doc_ids[prev_batch_idx]))
                        writer.addDocument(doc)
                        doc = Document()
                    doc.add(
                        FloatDocValuesField(
                            to_field_name(term.item()),
                            emb_batch[batch_idx, term].item(),
                        )
                    )
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


if __name__ == "__main__":
    config = Config(model_type=ModelType.K_SPARSE, dataset="msmarco_300000")
    print("Device:", config.device, torch.cuda.is_available())
    print("Torch:", torch.__version__)

    wandb_model_name = "model-hzq51uqh:v0"
    run = wandb.init()
    artifact = run.use_artifact(
        f"vector-search/{config.project}/{wandb_model_name}", type="model"
    )
    artifact_dir = artifact.download()
    model = Autoencoder.load_from_checkpoint(
        f"artifacts/{wandb_model_name}/model.ckpt"
    ).to(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_model_id, use_fast=True)
    model.eval()
    model.freeze()
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False
    encode_sparse_from_docs = build_encode_sparse_fun(
        config=config, tokenizer=tokenizer, model=model, threshold=None
    )
    print(encode_sparse_from_docs("test").shape)
    print("Number of nonzero", torch.count_nonzero(encode_sparse_from_docs("test")))

    runner = LuceneRunner(encode_fun=encode_sparse_from_docs)
    runner.delete_index()
    runner.index(batch_size=200)

    start = time.time()
    search_results = runner.search(top_k=1000)
    print("Search time:", time.time() - start)

    ndcg, _map, recall, precision, mrr = eval_with_dot_score_function(
        qrels=runner.qrels, results=search_results
    )
    print(ndcg, _map, recall, precision, mrr)
    start = time.time()
    runner.queries = {1: "Some test query"}
    runner.search()
    print("Query time:", time.time() - start)
