import pyperf
from scipy import stats
import numpy as np
import sys
import gc
import time


def run_bench(bench_name, func_to_bench, confidence=0.95):
    runner = pyperf.Runner()
    bench = runner.bench_func(bench_name, func_to_bench)
    if "--worker" in sys.argv:
        return
    values = bench.get_values()
    ci_low, ci_high = stats.t.interval(
        confidence, len(values) - 1, loc=np.mean(values), scale=stats.sem(values)
    )
    print(f"Confidence interval ({confidence}): [{ci_low}, {ci_high}]")


def run_bench_custom(
    bench_name,
    func_to_bench,
    confidence=0.95,
    warmup=5,
    repeats=50,
    min_total_s=None,
    unit="ms",
    rng_seed=42,
):
    for _ in range(warmup):
        func_to_bench()

    was_gc = gc.isenabled()
    gc.disable()
    samples = []
    try:
        if min_total_s is not None:
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) < float(min_total_s) or len(samples) < repeats:
                s0 = time.perf_counter()
                func_to_bench()
                samples.append(time.perf_counter() - s0)
        else:
            for _ in range(repeats):
                s0 = time.perf_counter()
                func_to_bench()
                samples.append(time.perf_counter() - s0)
    finally:
        if was_gc:
            gc.enable()

    s = np.array(samples, float)

    p50, p90, p95 = np.quantile(s, [0.5, 0.9, 0.95])

    ci = stats.bootstrap(
        (s,),
        np.median,
        confidence_level=confidence,
        n_resamples=min(50_000, max(5_000, 200 * len(s))),
        random_state=rng_seed,
        vectorized=False,
    )
    lo, hi = ci.confidence_interval.low, ci.confidence_interval.high

    scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit]

    def f(x):
        return f"{x * scale:.3f} {unit}"

    print(f"{bench_name}: n={len(s)}")
    print(f"median: {f(p50)}  (CI {int(confidence * 100)}%: [{f(lo)}, {f(hi)}])")
    print(f"p90:    {f(p90)}   p95: {f(p95)}")
    print(f"mean:   {f(s.mean())}  std: {f(s.std(ddof=1))}")



# if __name__ == "__main__":
#
#     def heavy_func(n):
#         prev, curr = 0, 1
#         for _ in range(n - 1):
#             prev, curr = curr, prev + curr
#
#     run_bench("Fibbo", lambda: heavy_func(1000))
