from common.bench import run_bench, calc_stats
from common.datasets import load_dataset
from common.lucene_inverted_index import LuceneInvertedIndex, func_to_bench, to_terms_and_scores
from spar_k_means_bert.util.eval import eval_with_dot_score_function
from transformers import AutoModelForMaskedLM, AutoTokenizer
import argparse
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--eval-or-bench", type=str, default="eval", help="eval or bench (default eval)"
)
parser.add_argument('-d', '--dataset', type=str, default='msmarco', help='BEIR dataset name (default msmarco)')
parser.add_argument(
    "-l",
    "--dataset-length",
    type=int,
    default=None,
    help="Dataset length (default None, all dataset)",
)
args = parser.parse_args()
print(f"Params: {args}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "rasyosef/splade-mini"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)


def encode_sparse(docs):
    tokens = tokenizer(docs, return_tensors="pt", padding=True, truncation=True).to(device)
    output = model(**tokens)
    # aggregate the token-level vecs and transform to sparse
    vecs = (
        torch.max(
            torch.log(1 + torch.relu(output.logits))
            * tokens.attention_mask.unsqueeze(-1),
            dim=1,
        )[0]
    )
    return vecs


corpus, queries, qrels = load_dataset(dataset=args.dataset, split="dev", length=args.dataset_length)
print(f"Corpus size={len(corpus)}, queries size={len(queries)}, qrels size={len(qrels)}")
inverted_index = LuceneInvertedIndex("./runs/common/lucene_splade_index")
if inverted_index.size() == 0:
    batch_size=8
    corpus = {
        doc_id: (doc["title"] + " " + doc["text"]).strip()
        for doc_id, doc in corpus.items()
    }
    inverted_index.index_all(corpus, batch_size, encode_sparse)
print("Inverted index size:", inverted_index.size())

if args.eval_or_bench == "eval":
    start = time.time()
    search_results = inverted_index.search(
        queries=queries,
        sparse_vector_calculator=lambda query: to_terms_and_scores(
            encode_sparse(query)
        ),
        top_k=1000,
    )
    print("Search time:", time.time() - start)

    ndcg, _map, recall, precision, mrr = eval_with_dot_score_function(
        qrels=qrels, results=search_results
    )
    print(ndcg, _map, recall, precision, mrr)
else:
    reader, searcher = inverted_index.get_reader_and_searcher()
    try:
        # Warmup
        warmup_query = "warmup benchmark query for measuring search latency"
        run_bench(
            func_to_bench=lambda: func_to_bench(
                inverted_index, searcher, warmup_query, encode_sparse,
            ),
            warmup=1000,
            repeats=10,
        )
        # Bench
        queries = list(queries.values())
        repeats = max(2000, len(queries)) // len(queries)
        samples = []
        for query in queries:
            query_samples = run_bench(
                func_to_bench=lambda: func_to_bench(
                    inverted_index, searcher, query, encode_sparse,
                ),
                warmup=0,
                repeats=repeats,
            )
            samples.extend(query_samples)
        calc_stats("Bench results", samples)
    finally:
        reader.close()
