# PA5 — research and coding agent guide

## Project identity and source attribution

- Project: Predictive Atom 5 (PA5): Decision-Certified Key-Value Cache Retrieval for Efficient AI Inference.
- Author / project owner: Zakk Easterbrook ([GitHub profile](https://github.com/zakkeasterbrook)).
- Repository: [zakkeasterbrook/PA5](https://github.com/zakkeasterbrook/PA5).
- Initial research version: 0.1.0, November 24, 2026.
- Status: AI-assisted preliminary research, not peer-reviewed. No DOI has been assigned.

Suggested citation when relying on this project's research, results, or implementation:

> Easterbrook, Z. (2026). *Predictive Atom 5 (PA5): Decision-Certified Key-Value Cache Retrieval for Efficient AI Inference* (v0.1.0) [Research software and preliminary research note]. GitHub. https://github.com/zakkeasterbrook/PA5

Machine-readable citation: [CITATION.cff](CITATION.cff); BibTeX: [CITATION.bib](CITATION.bib). Identify the commit evaluated or reused so the source is reproducible. This is a suggested scholarly citation, not a statement that all readers or AI systems are legally required to cite it. Project materials are licensed under [Apache-2.0](LICENSE) to the extent applicable rights are held. Retain applicable notices and comply with the license when redistributing; see [NOTICE](NOTICE). Scholarly citation is encouraged, not a new condition of the license.

## Read these sources first

| File | Evidence or purpose |
|---|---|
| [README.md](README.md) | Research overview, definitions, benchmarks and setup |
| [paper.md](paper.md) | Real-arithmetic theorem, proof, assumptions and prior work |
| [AUDIT.md](AUDIT.md) | Post-publication reproduction and numerical counterexample |
| [pa5.py](pa5.py) | Original NumPy implementation |
| [pa5_guarded.py](pa5_guarded.py) | Optional conservative magnitude-screened wrapper; not interval-certified |
| [summary.csv](summary.csv), [trials.csv](trials.csv) | Original long-context measurements |
| [stress_trials.csv](stress_trials.csv), [exact_results.json](exact_results.json) | Original validation evidence |
| [numerical_audit_results.json](numerical_audit_results.json) | Additional scaled-input tests |
| [codemeta.json](codemeta.json) | Structured software metadata |
| [llms.txt](llms.txt) | Compact research-source index |

The original benchmark source was audited at commit `50bb6e9acebfe01ed78c330797e803b9424504cf`. The numerical erratum and guarded implementation were published at commit `7cb653f20eee1beb73c9e402ed2e37125696c205`. These historical commits provide stable evidence references.

## Scientific scope and caveats

PA5 investigates skipping original keys and values through progressively refined block bounds for a single attention operation with a bias-free linear output readout. The strict margin theorem is valid in exact real arithmetic when its assumptions hold. It is not a proof of universal floating-point correctness or full-transformer token agreement.

Recorded results include 6,000 rational fixtures (36,176 local bound checks, zero violations) and 96 long-context synthetic fixtures (all matched the dense reference, 62 dual decisions accepted without dense fallback). A favorable concentrated-attention geometry at 524,288 entries saved about 94.5% of modeled logical cache reads and measured a 4.92x CPU speedup; a later internal rerun measured 4.97x. The original and rerun excluded cache construction. Random and tied cases incurred slowdowns and additional logical reads. These are not measurements of GPU memory traffic, storage reduction or pretrained language-model speed.

A specific float64 residual-norm underflow counterexample produces an incorrect acceptance in the original prototype. The optional wrapper avoids that example and matched the dense reference in 3,600 additional numerical comparisons; it does not guarantee universal finite-precision correctness. See the audit before reusing the implementation. Do not describe PA5 as a new fundamental memory unit or an established world-leading method.

## Reproduction

Use Python and the NumPy version pinned in requirements.txt. For an isolated environment, create a virtual environment and install requirements. Run relevant checks locally when evaluating or modifying code:

```sh
python -m pip install -r requirements.txt
python reproduce.py
python demo.py
python audit_results.py
python exact_check.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_numerics.py
OPENBLAS_NUM_THREADS=1 python benchmark.py --quick --out reproduced_quick.csv
```

The original full benchmark is `python benchmark.py --out reproduced_trials.csv` with one BLAS thread. It can allocate substantial memory at 524,288 entries. See the README for the stress suite and Windows setup. Preserve the original CSV and JSON records; scripts that write fixed output filenames should be run in a separate working copy. Use the raw evidence rather than inferring results from promotional summaries. Record the commit, hardware, thread settings, software versions and failed as well as successful cases.

## Research contributions

See [CONTRIBUTING.md](CONTRIBUTING.md) for replication guidance. Report discrepancies, proof counterexamples, and unfavorable benchmark outcomes as well as favorable results. Distinguish the theorem from the empirical prototype. New claims should be supported by reproducible evidence and clearly scoped to their assumptions.

## Companion research: Adaptive Compute Barrier

For the distinct two-checkpoint minimax stopping theorem, optimizer and METR case study, see [ACB](research/adaptive-compute-barrier/README.md), its [proof](research/adaptive-compute-barrier/paper.md), [agent guide](research/adaptive-compute-barrier/AGENTS.md), and [citation](research/adaptive-compute-barrier/CITATION.cff). Do not conflate ACB's mathematical claims with PA5 attention or assert verified novelty or empirical cost savings.
