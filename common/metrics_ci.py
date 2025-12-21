import argparse
import json
from typing import Dict, Tuple

import numpy as np

from common.datasets import load_dataset


def _results_for_qrels(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    return {qid: results.get(qid, {}) for qid in qrels}


def _top_k_results(
    results: Dict[str, Dict[str, float]],
    k: int,
) -> Dict[str, Dict[str, float]]:
    top_results = {}
    for qid, doc_scores in results.items():
        if len(doc_scores) <= k:
            top_results[qid] = doc_scores
            continue
        ranked = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        top_results[qid] = dict(ranked)
    return top_results


def per_query_ndcg_mrr(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    if k < 1:
        raise ValueError("k must be >= 1")
    try:
        import pytrec_eval
    except ImportError as exc:
        raise ImportError("pytrec_eval is required to use BEIR evaluation.") from exc

    results_for_eval = _results_for_qrels(qrels, results)

    ndcg_metric = f"ndcg_cut.{k}"
    ndcg_key = f"ndcg_cut_{k}"
    ndcg_eval = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_metric})
    ndcg_scores_by_qid = ndcg_eval.evaluate(results_for_eval)
    ndcg_scores = [
        ndcg_scores_by_qid.get(qid, {}).get(ndcg_key, 0.0)
        for qid in qrels
    ]

    mrr_eval = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    mrr_scores_by_qid = mrr_eval.evaluate(_top_k_results(results_for_eval, k))
    mrr_scores = [
        mrr_scores_by_qid.get(qid, {}).get("recip_rank", 0.0)
        for qid in qrels
    ]

    return np.asarray(ndcg_scores, dtype=float), np.asarray(mrr_scores, dtype=float)


def bootstrap_mean_ci(
    values: np.ndarray,
    num_samples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 1234,
) -> Tuple[float, float, float]:
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    if num_samples < 2:
        raise ValueError("num_samples must be >= 2")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in the interval (0, 1)")
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = np.empty(num_samples, dtype=float)
    for idx in range(num_samples):
        samples[idx] = rng.choice(values, size=n, replace=True).mean()
    alpha = (1 - confidence) / 2
    mean = float(np.mean(values))
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1 - alpha))
    return mean, lo, hi


def estimate_ndcg_mrr_ci(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k: int = 10,
    num_samples: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 1234,
) -> Dict[str, Dict[str, float]]:
    ndcg_scores, mrr_scores = per_query_ndcg_mrr(qrels, results, k=k)
    ndcg_mean, ndcg_lo, ndcg_hi = bootstrap_mean_ci(
        ndcg_scores, num_samples=num_samples, confidence=confidence, seed=seed
    )
    mrr_seed = None if seed is None else seed + 1
    mrr_mean, mrr_lo, mrr_hi = bootstrap_mean_ci(
        mrr_scores, num_samples=num_samples, confidence=confidence, seed=mrr_seed
    )
    return {
        f"ndcg@{k}": {"mean": ndcg_mean, "ci_low": ndcg_lo, "ci_high": ndcg_hi},
        f"mrr@{k}": {"mean": mrr_mean, "ci_low": mrr_lo, "ci_high": mrr_hi},
    }


def _print_ci(label: str, stats: Dict[str, float], confidence: float) -> None:
    ci_pct = int(confidence * 100)
    print(
        f"{label}: mean={stats['mean']:.6f} "
        f"CI{ci_pct}%=[{stats['ci_low']:.6f}, {stats['ci_high']:.6f}]"
    )


def _load_results(results_path: str) -> Dict[str, Dict[str, float]]:
    with open(results_path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, required=True, help="results.json path")
    parser.add_argument("--dataset", type=str, default="scifact", help="BEIR dataset name")
    parser.add_argument("--split", type=str, default="test", help="BEIR split name")
    parser.add_argument("--dataset-length", type=int, default=None, help="Dataset length")
    parser.add_argument("--k", type=int, default=10, help="k for ndcg@k and mrr@k")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap samples")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    args = parser.parse_args()

    _, _, qrels = load_dataset(
        dataset=args.dataset, split=args.split, length=args.dataset_length
    )
    results = _load_results(args.results_path)

    stats = estimate_ndcg_mrr_ci(
        qrels,
        results,
        k=args.k,
        num_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )
    _print_ci(f"ndcg@{args.k}", stats[f"ndcg@{args.k}"], args.confidence)
    _print_ci(f"mrr@{args.k}", stats[f"mrr@{args.k}"], args.confidence)
