import gc
import time
from typing import Callable, Dict, Optional

import numpy as np
from scipy import stats


def run_bench(
    bench_name: str,
    func_to_bench: Callable[[], None],
    confidence: float = 0.95,
    warmup: int = 5,
    repeats: int = 100,
    min_total_s: Optional[float] = None,
    unit: str = "ms",
) -> Dict[str, float]:
    valid_units = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    if unit not in valid_units:
        raise ValueError(f"Unknown unit: {unit}. Expected one of {sorted(valid_units.keys())}")
    if not (0 < confidence <= 1):
        raise ValueError("confidence must be in the interval (0, 1]")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    for _ in range(warmup):
        func_to_bench()

    was_gc = gc.isenabled()
    gc.disable()
    samples: list[float] = []
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

    samples = np.array(samples, dtype=float)
    n = len(samples)

    p50, p90, p95 = np.quantile(samples, [0.5, 0.9, 0.95])
    mean = np.mean(samples)
    std = samples.std(ddof=1)
    sem = std / np.sqrt(n)

    lo, hi = stats.norm.interval(confidence, loc=mean, scale=sem)

    scale = valid_units[unit]
    def fmt(x: float) -> str:
        return f"{x * scale:.3f} {unit}"

    print(f"{bench_name}: n={n}")
    print(f"mean:   {fmt(mean)}   std:    {fmt(std)}   CI {int(confidence * 100)}%: [{fmt(lo)}, {fmt(hi)}]")
    print(f"p50:    {fmt(p50)}   p90:    {fmt(p90)}   p95: {fmt(p95)}")



if __name__ == "__main__":

    def heavy_func(n):
        prev, curr = 0, 1
        for _ in range(n - 1):
            prev, curr = curr, prev + curr

    run_bench("Fibbo", lambda: heavy_func(1000))
