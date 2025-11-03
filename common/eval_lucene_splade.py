from common.bench import run_bench, calc_stats
from common.datasets import load_dataset
from spar_k_means_bert.util.eval import eval_with_dot_score_function
from sparsifier_model.run import LuceneRunner
from transformers import AutoModelForMaskedLM, AutoTokenizer
import argparse
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--eval-or-bench", type=str, default="eval", help="eval or bench (default eval)"
)
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

model_id = "naver/splade-cocondenser-ensembledistil"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)


def encode_sparse(docs):
    tokens = tokenizer(docs, return_tensors="pt", padding=True, truncation=True)
    output = model(**tokens)
    # aggregate the token-level vecs and transform to sparse
    vecs = (
        torch.max(
            torch.log(1 + torch.relu(output.logits))
            * tokens.attention_mask.unsqueeze(-1),
            dim=1,
        )[0]
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    return vecs


runner = LuceneRunner(
    encode_fun=encode_sparse,
    dataset="msmarco",
    docs_number=args.dataset_length,
    index_path="./runs/common/lucene_splade_index",
)
if runner.size() == 0:
    runner.delete_index()
    runner.index(batch_size=128)
print("Inverted index size:", runner.size())

if args.eval_or_bench == "eval":
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
else:
    reader, searcher = runner.get_reader_and_searcher()
    try:
        # Warmup
        run_bench(func_to_bench=lambda: runner.search_by_query(searcher, "warmup benchmark query for measuring search latency"), warmup=1000, repeats=10)
        # Bench
        queries = list(runner.queries.values())
        repeats = max(2000, len(queries)) // len(queries)
        samples = []
        for query in queries:
            query_samples = run_bench(func_to_bench=lambda: runner.search_by_query(searcher, query), warmup=0, repeats=repeats)
            samples.extend(query_samples)
        calc_stats("Bench results", samples)
    finally:
        reader.close()
