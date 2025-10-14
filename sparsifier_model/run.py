import time
import itertools
import sys
from transformers import AutoTokenizer, AutoProcessor
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
from sparsifier_model.util.dataset import get_dataloader
from datasets import load_dataset


class LuceneRunner:
    def __init__(self, encode_fun, dataset=None, docs_number=None, modality="image2image"):
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
        self.modality = modality
        if modality == "image2image":
            corpus = load_dataset("nlphuji/flickr30k", cache_dir="/opt/dlami/nvme/tmp/hf_cache")['test']
            self.corpus = {img_id:image for img_id, image in zip(corpus['img_id'],corpus['image'])}

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
            writer.forceMerge(1, True)
            writer.commit()
            writer.close()

    def size(self):
        try:
            reader = DirectoryReader.open(FSDirectory.open(self.index_jpath))
            num_docs = reader.numDocs()
            reader.close()
            return num_docs
        except:
            return 0

    def search(self, top_k=10):
        reader = DirectoryReader.open(FSDirectory.open(self.index_jpath))
        searcher = IndexSearcher(reader)
        results = {}

        try:
            query_id = "0"
            if self.modality == "image2image":
                img = self.corpus[query_id]
                query_emb = self.encode([img], self.modality)[0]
            elif self.modality == "text2image":
                query_emb = self.encode(["water"], self.modality)[0]
            else:
                raise ValueError(f"Unknown modality: {self.modality}")
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
    model_id = "openai/clip-vit-base-patch32"
    config=Config(
        backbone_model_id=model_id,
        k=100,
        model_type=ModelType.K_SPARSE,
        dataset="flickr30k")
    print("Device:", config.device, torch.cuda.is_available())
    print("Torch:", torch.__version__)

    wandb_model_name = "model-eahrdhhm:v0"
    run = wandb.init()
    artifact = run.use_artifact(
        f"vector-search/{config.project}/{wandb_model_name}", type="model"
    )
    artifact_dir = artifact.download()
    model = Autoencoder.load_from_checkpoint(
        f"artifacts/{wandb_model_name}/model.ckpt"
    ).to(config.device)

    model.eval()
    model.freeze()
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False
    def encode_sparse_from_docs(docs, modality):
        docs = list(docs)
        return model.encode(docs, modality)
    #corpus = load_dataset("nlphuji/flickr30k", cache_dir="/opt/dlami/nvme/tmp/hf_cache")['test'][1]
    #print(encode_sparse_from_docs([corpus['image']]).shape)
    #print("Number of nonzero", torch.count_nonzero(encode_sparse_from_docs([corpus['image']])))

    runner = LuceneRunner(encode_fun=encode_sparse_from_docs)
    if runner.size() == 0:
        runner.delete_index()
        runner.index(batch_size=128)
    print("Inverted index size:", runner.size())

    #start = time.time()
    search_results = runner.search()
    print(search_results)
    #print("Search time:", time.time() - start)

    #ndcg, _map, recall, precision, mrr = eval_with_dot_score_function(
    #    qrels=runner.qrels, results=search_results
    #)
    #print(ndcg, _map, recall, precision, mrr)
    #start = time.time()
    #runner.queries = {1: "Some test query"}
    #runner.search()
    #print("Query time:", time.time() - start)
