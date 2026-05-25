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
