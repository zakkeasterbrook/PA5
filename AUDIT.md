# PA5 post-publication reproducibility and numerical audit

**November 24, 2026 · AI-assisted internal replication, not independent peer review**

Audited commit: [50bb6e9](https://github.com/zakkeasterbrook/PA5/commit/50bb6e9acebfe01ed78c330797e803b9424504cf). The published original Python algorithm, test scripts, recorded JSON, dependency file, and original CSVs were verified against their Git blob hashes before executing a fresh copy of the archived source. The original experiments and algorithm remain unchanged.

## Reproduction

Environment: AMD EPYC 9V74, Python 3.13.5, NumPy 2.3.5, one BLAS/OpenMP thread for the benchmarks.

- Exact rational tests: 6,000 fixtures, 36,176 local bounds, 2,027 accepted decisions, zero violations. Result JSON matched the original byte-for-byte.
- Original stress suite: all 4,000 decisions agreed with the dense reference; 1,895 accepted decisions. JSON and per-trial CSV matched the original byte-for-byte.
- Full CPU benchmark: all 288 fixture-mode comparisons agreed; 62/96 dual-mode decisions accepted. All 288 per-trial records matched the original in every non-timing field.
- The identical-key quickstart selected output index 14 and read 0.805664% of modeled raw cache bytes; no original KV blocks retrieved.

New measurements at 524,288 entries, with four synthetic fixtures per geometry:

| Geometry | Mean logical reads / dense | Median dense ms | Median PA5 ms | Ratio | Median cache construction ms |
|---|---:|---:|---:|---:|---:|
| Identical keys | 0.00805664 | 14.126 | 0.691 | 20.45x | 136.097 |
| Concentrated | 0.05493164 | 13.289 | 2.671 | 4.97x | 135.154 |
| Outlier | 0.10180664 | 15.308 | 4.911 | 3.12x | 127.753 |
| Clustered | 0.63305664 | 13.500 | 22.171 | 0.61x | 145.275 |
| Random | 1.47680664 | 12.989 | 36.409 | 0.36x | 130.830 |
| Tied | 1.47680664 | 13.452 | 36.824 | 0.37x | 137.723 |

Timing excludes cache construction. On the concentrated fixtures, simple division of the median construction time by the median per-query savings suggests approximately 13 repeat queries to amortize construction on this hardware. This is illustrative arithmetic, not an observed multi-query or end-to-end benchmark. Logical read estimates are not DRAM traffic measurements.

## Numerical erratum: the original finite-precision margin test can accept an incorrect decision

The following finite float64 data reproduce an incorrect acceptance:

```python
import numpy as np
from pa5 import Cache, dense
K = np.array([[-1.], [1.]])
V = np.array([[2e-200], [-1e-200]])
q = np.array([1.])
W = np.array([[1e150, -1e150]])
print(dense(K, V, q, W))          # 1
print(Cache(K, V, 2).decide(q,W)) # token 0, accepted True, bound 0
```

The squared centered residuals underflow in float64, leading to a computed norm of zero even though the residual is nonzero. The approximate block score has the opposite sign from the exact weighted sum. Indeed, in real arithmetic the latter is proportional to 2 exp(-1) - exp(1), which is negative. Thus the exact winner is index 1. This does not contradict the real-arithmetic theorem, which requires valid bounds. It disproves any universal floating-point correctness claim for the original prototype.

## Optional guarded path

[pa5_guarded.py](pa5_guarded.py) adds conservative raw-magnitude screening (nonzero entries must be between 1e-50 and 1e50 for the original margin test), dimension validation, an owned read-only cache copy, and finite-score checks in the range-screened dense fallback. The explicit counterexample returns the reference result without accepting the approximate decision. The wrapper is **not** a rigorous interval implementation: it still invokes the original numerical path on screened inputs; intermediate arithmetic, cancellation, rounding and dense-reference accuracy are not universally certified. Its performance has not been benchmarked. Original results are for pa5.py, not the wrapper.

The new regression test uses 1,800 seeded scaled-input fixtures with two modes each (3,600 evaluations). Original and guarded paths agreed with the floating-point dense reference in all of these random evaluations, separately from the explicit original failure. The guarded path accepted 331 evaluations and forced a range-screened fallback in 3,204; remaining cases used the original margin/refinement path and fallback. The extra mutation and overflowing-logit checks passed. These tests do not establish correctness for all inputs.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_numerics.py
```

[Recorded regression results](numerical_audit_results.json).

## Verdict

The restricted real-arithmetic result stands, and the original favorable and unfavorable synthetic benchmark outcomes reproduce. A finite-precision counterexample is now documented, along with a conservative experimental mitigation. No claim of fully certified floating-point execution, pretrained LLM acceleration, storage reduction, energy savings or superiority to optimized attention implementations is supported. A rigorous numerical enclosure and real-model evaluation remain necessary.
