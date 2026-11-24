# Predictive Atom 5 (PA5)

**Decision-certified dual key/value retrieval for attention with a linear readout**  
Zakk Easterbrook · November 24, 2026 · Preliminary research note, v0.1.0

PA5 asks whether an attention calculation can preserve its winning output without reading every original key or value. It uses block statistics, two analytic error bounds, progressive key/value refinement, and a dense fallback. Original KV is retained: the goal is reducing logical access, not storage capacity. This is an AI-assisted research project; the theory and experiments are available for independent scrutiny, not peer-reviewed or claimed to be state of the art.

## Reproduce

Python 3.13.5 and NumPy 2.3.5 were used. On a POSIX shell:

```sh
python -m pip install -r requirements.txt
python exact_check.py
OPENBLAS_NUM_THREADS=1 python stress.py
OPENBLAS_NUM_THREADS=1 python benchmark.py --quick --out quick.csv
OPENBLAS_NUM_THREADS=1 python benchmark.py --out trials.csv
```

The full benchmark uses up to 524,288 rows of 32-dimensional float64 K and V (256 MiB raw combined at maximum length), plus working memory. Use --quick for smaller machines. Run on an idle host; timings vary. Do not mutate the arrays after Cache construction. Cache accepts full equal-size blocks only. q must already include attention scaling. Tests use a fixed bias-free linear output readout, not a general transformer.

## What happened

- 6,000 exact-rational fixtures: 36,176 local bound checks, 2,027 accepted unique decisions, 1,200 forced ties, zero violations.
- 96 large synthetic fixtures across 4 context lengths and 6 geometries: all 288 dense/dual/value-only evaluations matched dense. Of the 96 dual evaluations, 62 accepted (see summary.csv); the rest used exact dense fallback.
- 2,000 additional stress fixtures: 4,000 dual/value-only evaluations, 1,895 accepted, zero disagreements.

At 524,288 entries (four fixtures per geometry):

| Geometry | Dual logical reads / dense | Median dual ms | Median dense ms | Ratio dense/dual |
|---|---:|---:|---:|---:|
| Flat | 0.00806 | 0.724 | 11.928 | 16.47x |
| Concentrated | 0.05493 | 2.423 | 11.926 | 4.92x |
| Outlier | 0.10181 | 4.818 | 12.269 | 2.55x |
| Clustered | 0.63306 | 20.301 | 12.032 | 0.59x |
| Random | 1.47681 | 33.388 | 11.785 | 0.35x |
| Tie | 1.47681 | 34.814 | 11.479 | 0.33x |

These are deliberately constructed fixtures, not real LLM KV caches. Timing is the median of per-fixture medians of three calls, single-threaded NumPy, excluding summary construction. Logical reads include metadata and repeated dense fallback reads, not measured DRAM traffic, workspace, model weights, or cache-construction reads. The flat case is a degenerate best case, not a general speed claim. No GPU, energy, capacity, or full-model accuracy claim is supported. The implementation has a numerical safety cushion, not a rigorously rounded certificate; the real-arithmetic proof is separate. The value-only comparison is the same prototype with all keys opened upfront, not an optimized published competitor.

## Contents

- [Research note and proof](paper.md)
- [Reference implementation](pa5.py)
- [Exact rational tests](exact_check.py), [results](exact_results.json)
- [Adversarial test](stress.py), [results](stress_results.json)
- [Long-context benchmark](benchmark.py), [summary](summary.csv)
- [Citation metadata](CITATION.cff)

The scripts regenerate per-trial CSV files locally. Raw original trial logs are preserved in the accompanying downloadable research archive; this GitHub version contains the aggregate measurements and reproducibility scripts.

## Open questions

Test on pretrained multihead models; propagate bounds through normalization, residuals, nonlinear layers and output biases; use certified floating-point enclosures; compare with optimized FlashAttention and current adaptive KV methods on identical workloads; design an early-abort policy for diffuse/tied cases; implement incremental immutable cache summaries and GPU kernels. Prior-art overlap is substantial; the narrow evaluated mechanism is the two-stage block error rule for an affine argmax, not the invention of adaptive KV compression or a new physical unit of memory.
