# Independent replication and feedback

PA5 is a preliminary research prototype. Independent replications, counterexamples, proof criticism, numerical edge cases and unfavorable performance reports are welcome. A software license has not yet been selected; contact the repository owner about reuse or redistribution permissions.

## Report a result

Use a GitHub issue if issues are enabled. Include:

- The exact Git commit, Python/NumPy version, OS, CPU or GPU, BLAS implementation and thread count.
- Command line, seed, context length, block size, key/value dimensions, and output count.
- Baseline implementation and operation order; specify timing warmup and whether summary construction is included.
- Observed decision agreement, acceptance count, fallback rate, logical reads and runtime, with per-trial data if possible.
- Any failures and a minimal reproducer. Distinguish a real-arithmetic counterexample from a floating-point or implementation issue.

Do not attach proprietary model traces, credentials, private prompts, or other data you lack permission to share.

## Research questions

1. Can the bound be tightened or made cheaper while preserving its proof assumptions?
2. Can numerical enclosures certify the finite-precision implementation?
3. When does preprocessing and refinement overhead outweigh retrieval savings?
4. How does PA5 compare with optimized dense and selective KV methods on real-model traces?
5. Can the decision argument be extended beyond the bias-free linear readout?

Report all conditions, not just favorable ones. The current work makes no end-to-end language model, GPU speed, or world-first claim.
