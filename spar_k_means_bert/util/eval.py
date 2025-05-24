import json
from typing import Dict
from beir.retrieval.evaluation import EvaluateRetrieval
from common.datasets import load_dataset


def eval_with_dot_score_function(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
):
    retriever = EvaluateRetrieval(score_function="dot")
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values
    )
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    return ndcg, _map, recall, precision, mrr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', type=str, help='results.json path')
    args = parser.parse_args()
    print(f"Params: {args}")

    with open(args.results_path, "r", encoding="utf-8") as f:
        results: Dict[str, Dict[str, float]] = json.loads(f.read())

    corpus, queries, qrels = load_dataset(dataset="scifact", length=None)
    ndcg, _map, recall, precision, mrr = eval_with_dot_score_function(qrels, results)
    print("=============")
    print(ndcg, _map, recall, precision, mrr)
    print("=============")
