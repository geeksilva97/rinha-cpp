# Benchmark results

VM: GCP `n1-custom-4-8192`, Haswell, 4 vCPU, 8 GB RAM, pd-standard.
This matches the rinha-2026 reference machine (Mac mini Late 2014).

k6 stress: ramping-arrival-rate 1 → 900 RPS over 120s, 54 100 requests
total (mix of fraud + legit, with edge cases).

## Sweep

| run               | NPROBE | FAST_NPROBE | score   | p99     | fp | fn | notes                       |
|-------------------|-------:|------------:|--------:|--------:|---:|---:|-----------------------------|
| baseline          |     70 |           5 | 5654.8  | 1.37 ms |  1 |  1 | adaptive nprobe             |
| nprobe-100        |    100 |          10 | 5237.7  | 3.82 ms |  0 |  1 | wider search hurts p99 more |
| fast-3            |     70 |           3 | 5495.1  | 1.37 ms |  1 |  5 | too-shallow first pass      |
| **prefault**      |     70 |           5 | **5664.5** | **1.34 ms** | 1 | 1 | madvise prefault on boot    |
| always-full       |     70 |          70 | 4375.0  | 27.8 ms |  0 |  1 | no fast path = burst killer |

Baseline of the previous Ruby/C++ submission on the same test:
**score 4362.78, p99 18.88 ms, fp=6 fn=3.**

The C++ rewrite gains +1302 points, almost all from latency
(p99 18.88 ms → 1.34 ms ≈ +1276 points by the rinha formula).

## Score decomposition (prefault run)

```
score_p99 = 1000 · log10(1000 / 1.34) = 2873
E         = 1·fp + 3·fn + 5·err = 1·1 + 3·1 + 0 = 4
ε         = 4 / 54100 ≈ 7.4e-5 → clamped to 0.001
score_det = 1000·log10(1/0.001) − 300·log10(1+4)
          = 3000 − 209.7 = 2790
final     = 2873 + 2790 = 5664   (matches the k6 summary)
```

## Headroom

Total possible: 6000. Distance to ceiling: ~335 points, split into:

- **+127** by pushing p99 from 1.34 ms toward 1.0 ms. Hard. We're
  already a single read+parse+kNN call on a hot mmap'd index. Wins
  would have to come from container-level scheduling (CPU pinning,
  longer cgroup quota period), not user code.

- **+210** by eliminating the 2 remaining detection mistakes
  (fp=1, fn=1). always-full nprobe=70 only eliminates one, costing
  1300 p99 points — net very negative. The 2 errors look intrinsic
  to the IVF labels at the 0.6 threshold; reaching them likely
  requires retraining or a small re-ranker.

## Retraining experiment

The original `ivf.bin` was trained with k-means at `T=20` epochs with
no record of the final `move` value. We retrained from scratch with
`T=200` and `EPS_MOVE=1e-6` to see whether the older index was
undertrained and whether better-converged clusters would fix the two
remaining errors.

The training did converge much more tightly: `move` at iter 20 was
~3.6 (where the old training stopped); at iter 200 it had dropped to
**0.115** — about 30× tighter. But the resulting index performed
slightly *worse* on the test set:

| index            | training | fp | fn | total errors |
|------------------|---------:|---:|---:|-------------:|
| `ivf.bin`        | T=20     |  1 |  1 |  2 |
| `ivf.bin` (new)  | T=200    |  2 |  1 |  3 |

Why retraining didn't help: the eval tool runs a brute-force top-5
across all 3 M reference vectors for the disagreement cases. For
both the original FP1 and the original FN1, brute-force returns the
*same* count as full-nprobe — meaning the L2-nearest training points
in 14-dim feature space simply disagree with the test labels. Better
clustering reshuffles which points are picked at any given nprobe,
but it can't change *which training points are closest*. One of
the new clusterings happened to push a borderline previously-correct
case into the wrong fast/full bucket, adding FP2.

Conclusion: **keep the T=20 ivf.bin**. The remaining two errors are at
the feature-engineering level, not the index level. Fixing them would
require either a richer feature vector or a post-classifier on top of
the kNN votes — neither is in scope for this submission.

The training infrastructure (`src/training/k_means.cpp` with
pause/resume + `src/training/build_ivf.cpp` to materialize a
checkpoint into `ivf.bin`) stays in the repo for future re-training
work.
