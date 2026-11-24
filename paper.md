# Predictive Atom 5: Decision-certified dual key/value retrieval for attention

**Zakk Easterbrook — AI-assisted preliminary research note, 2026-11-24, v0.1.0**

## Abstract

We study a restricted decision problem: preserve the unique argmax of a bias-free linear readout following a single attention calculation while selectively avoiding reads of the original keys and values. A block descriptor stores key and value centroids, a maximum key radius, and a value residual Frobenius norm. Two deterministic real-arithmetic bounds enable progression from summaries to exact keys and finally exact values. A margin test either certifies a unique winner or abstains; dense evaluation is the fallback. Exact rational property testing found no violations in 36,176 checks; a separate numerical implementation matched its dense reference across 96 synthetic long-context fixtures and 4,000 additional stress evaluations. At 524,288 entries, the concentrated synthetic geometry required 5.49% of modeled dense cache reads and measured 4.92x reference speed. Diffuse and tied geometries were slower. These are not claims about full LLMs, GPU speed, or state-of-the-art performance.

## 1. Problem and assumptions

Let keys k_i and values v_i be finite real vectors, q a fixed scaled query, and W a fixed matrix with columns w_j. For any finite common shift s, define x_i = exp(q dot k_i - s) > 0, y = sum_i x_i v_i, Z = sum_i x_i > 0. The target is argmax_j (w_j dot y / Z). Because Z is positive and common to all output scores, this equals argmax_j w_j dot y. This simplification relies on a bias-free linear readout; it does not generally apply across nonlinearities or additive biases. Assume exact real arithmetic, immutable complete equal-size blocks, and finite metadata that bounds the underlying data. The algorithm retains all original KV, rather than deleting information.

## 2. Block bounds

For a block of b entries let c be the mean key, R = max_i ||k_i - c||_2, mu the mean value, and F^2 = sum_i ||v_i - mu||_2^2.

By Cauchy-Schwarz, every scaled score lies in [q dot c - R||q||_2, q dot c + R||q||_2]. Monotonicity of exp implies that the weights lie in [L,H] where L = exp(q dot c - R||q||_2 - s) and H = exp(q dot c + R||q||_2 - s).

**Stage 0 (both original K and V unread).** Set m = (L+H)/2, delta = (H-L)/2, and y_hat = b m mu. The bound is

$$E_0 = \delta\sqrt{b}\sqrt{b\|\mu\|_2^2+F^2}.$$

Proof: y - y_hat = sum_i (x_i-m)v_i. By Cauchy-Schwarz its norm is at most sqrt(sum_i (x_i-m)^2) sqrt(sum_i ||v_i||^2). The first factor is bounded by sqrt(b) delta. The second equals sqrt(b ||mu||^2 + F^2), by the centered sum-of-squares identity. This proves the bound, including the degenerate case delta=0.

**Stage 1 (exact keys, original values unread).** Compute the actual x_i, their mean x_bar, and y_hat = b x_bar mu. Then

$$E_1 = F\sqrt{\sum_i(x_i-\bar{x})^2}.$$

Proof: since sum_i(v_i-mu)=0, y-y_hat = sum_i (x_i-x_bar)(v_i-mu). Cauchy-Schwarz gives the claimed bound.

**Stage 2 (both opened).** Compute y exactly; E_2 = 0.

These bounds need not decrease at every refinement for every possible input; they are individually valid. The algorithm eventually uses exact dense fallback when its finite refinement budget is exhausted.

## 3. Decision certificate

For all blocks g, sum the approximate block vectors to obtain Y_hat and sum their respective valid bounds to obtain E. The triangle inequality implies ||y-Y_hat||_2 <= E. Choose c = argmax_j w_j dot Y_hat. If for every j != c,

$$ (w_c-w_j)^\top\hat{Y} > E\|w_c-w_j\|_2, $$

then c is the unique exact maximizer. Indeed the true pairwise difference is at least the approximate margin minus E times the readout-direction norm, which is strictly positive by hypothesis. This is a sufficient condition, not a necessary condition. A tie never passes the strict test. An abstention does not establish that the approximation is wrong.

The NumPy implementation evaluates these formulas with a numerical guard rather than directed rounding. Consequently the theorem in real arithmetic does not imply that every possible floating-point execution is certified. Rigorous enclosures for metadata, dot products, exponentials, and accumulated errors are future work. The dense fallback reproduces the specific reference operation order used here.

## 4. Method and evaluation

The prototype stores two means and two scalar bounds per block. A common upper shift is chosen from the block score bounds. The largest current error blocks are progressively opened in batches. If every competing score is separated by the certificate, it returns the winner; otherwise the identical dense reference is called after a bounded number of passes. The values-only ablation opens all keys upfront. Selection heuristics do not affect the real-arithmetic theorem. No formal speed advantage follows from the theorem.

- Exact tests: 6,000 seeded rational fixtures, using arbitrary positive rational weights within their intervals and outward rational square-root bounds. Both local inequalities and accepted unique decisions are tested; 2,027 were accepted, 1,200 were forced ties, and there were zero violations across 36,176 local checks. This tests the algebra, not IEEE-754 exp.
- Long-context tests: four context sizes (4,096, 65,536, 262,144, 524,288), six synthetic geometries, four fixtures each; 32 key and value dimensions, 16 outputs, blocks of 128. All 288 dense/dual/value-only fixture-mode evaluations agreed. 62/96 dual evaluations accepted without dense fallback. Single-threaded NumPy timing is the median of three calls per fixture and then the median across four fixtures. Preprocessing is excluded from reported query times but captured in the regenerated trial CSV.
- Stress tests: 2,000 seeded fixtures with random sizes, extreme outliers, identical and near-identical output columns, and flat keys. 4,000 dual/values-only evaluations; 1,895 accepted, zero disagreements. Results are empirical rather than exhaustive numerical proofs.

Logical access is counted as descriptor bytes plus original K/V bytes opened, including repeated reads on fallback. This is not a physical memory-traffic measurement and excludes working arrays, output readout and summary construction. Original data remains stored. At b=128 and key/value dimension 32, metadata alone is (2*32+2)/(128*64) = 0.00805664 of raw cache bytes. Identical keys permit an exact stage-0 representation, apart from numerical effects; this is a constructed best case.

See README.md and summary.csv for recorded results including regressions. At maximum length, concentrated attention saves 94.51% of logical reads with 4.92x speed relative to the dense NumPy reference. Random and tied cases use approximately 147.68% of dense reads and are slower. The test geometry is intentionally favorable in some cases; it is not a sample from a real model. Timing is hardware dependent.

## 5. Limits and research agenda

This is a single attention operation with a bias-free readout, not autoregressive inference on a pretrained transformer. Positive findings do not establish end-to-end token agreement, storage reduction, GPU acceleration, or energy savings. Normalization cannot simply be omitted for arbitrary downstream networks. Production requires incremental summary maintenance, floating-point enclosures, dimension and finite-range validation, a fallback-cost-aware policy, optimized kernels, and comparison with strong existing methods on identical real-model traces. No priority, patentability, or novelty guarantee is made. These are applications of established inequalities to a specific progressive decision mechanism.

## 6. Related work

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022. Exact IO-aware attention is an important optimized baseline.
- Tang et al., [QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference](https://proceedings.mlr.press/v235/tang24l.html), ICML 2024. Uses query-dependent key-page bounds for cache selection; demonstrates related prior art.
- Saha and Ye, [The I/O Complexity of Attention, or How Optimal is Flash Attention?](https://arxiv.org/abs/2402.07443), 2024. Addresses full-attention IO complexity, a different task from certifying a restricted output decision.

This list is not a comprehensive literature or patent search. A systematic comparison with adaptive KV retrieval and certified inference approaches is required before a public novelty claim.

## 7. Reproducibility and disclosure

Run the exact and numerical test scripts from the repository root following README.md. Seeds, code, and aggregate results are included. Per-trial outputs can be regenerated; original raw CSVs are separately preserved. This preliminary note, code, and experiments were developed with AI assistance at the direction of the project owner. No peer review or independent replication has been completed.
