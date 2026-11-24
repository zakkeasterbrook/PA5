# The Adaptive Compute Barrier (ACB)

**A sharp minimax boundary for score-aware AI inference budgets**  
Zakk Easterbrook · November 24, 2026 · Research v0.1.0 · AI-assisted preliminary research

[Full research note and proof](paper.md) · [Exact optimizer](exact_policy.py) · [Source data and assumptions](DATA_SOURCES.md) · [Citation](CITATION.cff)

## Abstract

Can an AI system use observed progress to decide when to stop computing and *guarantee* better savings than an independently randomized budget? For two fixed spending checkpoints, we derive a necessary and sufficient condition under a precisely specified ambiguity class: retained, monotone, grid-valued verified scores with known population means but otherwise unrestricted joint distributions. We construct a gain-tilted adversarial witness proving the impossibility direction, an affine randomized rule proving the strict-improvement direction, and an exact two-variable optimizer for the best worst-case policy. We also formulate a restricted multi-checkpoint extension using linear programming and rational Bellman certificates. A case study uses selected public METR mean-score data released on November 20, 2026. These results characterize what can be guaranteed from aggregate means alone, not the behavior of a particular agent with known run-level data.

> **Status:** mathematical research under stated assumptions; not peer-reviewed. No scientific priority, universal compute lower bound, deployed savings, or new model benchmark is claimed. The theorem is a candidate research contribution, not a verified world-first discovery.

## The main theorem

Let $X\le Y$ be retained scores on $G=\{0,1/K,\ldots,1\}$ at two fixed cost checkpoints. Assume only $\mathbb E[X]=\mu$, $\mathbb E[Y]=\nu$, with $0\le\mu<\nu\le1$. For a randomized stopping rule $f:G\to[0,1]$, maximize $\inf_P\mathbb E_P[f(X)]$ subject to $\sup_P\mathbb E_P[f(X)(Y-X)]\le\varepsilon$, where $P$ ranges over all such monotone joint laws and $0<\varepsilon<\nu-\mu$. An independent randomized budget has optimal guaranteed stop probability $\alpha=\varepsilon/(\nu-\mu)$.

Define $z=\mu/\nu$ and let $a,b$ bracket $z$ in the score grid. There is a strictly better robust score-dependent rule **if and only if**

$$ (z-a)(b-z)>z(1-z)(1-\nu). $$

In particular, $\nu\le1-1/K$ implies no strict robust gain from score-dependent stopping. For 21 objectives this sufficient no-gain threshold is $20/21\approx95.238\%$. Above it, improvement is possible but not guaranteed: the full criterion is necessary and sufficient. These are two-checkpoint statements for an ambiguity class based solely on means, not an impossibility claim for adaptive inference with richer observations or a known trajectory distribution.

The proof reduces the sharp worst-case gain moment to a two-point distribution on neighboring grid values, constructs an admissible gain-tilted marginal in the no-gain regime, and uses a bounded affine rule in the strict-gain regime. The exact optimizer then reduces the entire finite-grid policy problem to a two-variable rational linear program over clipped-affine rules. See [paper.md](paper.md) for the full derivation and boundary cases.

## Verification and public-data case study

| Check | Result |
|---|---:|
| Exact rational witness and policy checks | 25,344; no failed assertions |
| Separately formulated numerical LP comparisons | 1,785; no failed assertions |
| METR Earthborne Rangers selected checkpoint pairs | 2,353 |
| Strictly increasing pairs satisfying the two-checkpoint no-gain criterion | 2,262 |
| Equal displayed-mean pairs | 91 |
| Archived multistage feasible-policy certificates checked exactly | 2 / 2 |

The first two scripts were rerun before publication and their recorded result objects matched exactly. “Separately formulated” means distinct optimization formulations, not external scientific replication. The METR analysis assumes the displayed means are exact inputs to a common-cohort monotone 21-objective model; these assumptions are not established from aggregate data. No inference was purchased or rerun. For the tested one-percentage-point loss tolerance, the restricted multistage optimization found no meaningful improvement over the optimal independent budget on these selected curves. The archived rational certificates establish feasibility, not a global multistage optimum. Data attribution and caveats: [DATA_SOURCES.md](DATA_SOURCES.md).

### Constructed examples of a provable benefit

| Model problem | Independent budget | Exact adaptive optimum |
|---|---:|---:|
| Binary score: $\mu=3/5,\nu=9/10,\varepsilon=3/100$ | 10% | 64% |
| 21 objectives: $\mu=41/42,\nu=1,\varepsilon=1/84$ | 50% | 75% |

Percentages are guaranteed fractions of the additional checkpoint cap avoided, at identical worst-case mean-score loss; these are mathematical examples, not measured inference-cost savings.

## Reproduce

Python 3.13.5, NumPy 2.3.5 and SciPy 1.17.0 were used. No API key, GPU or paid inference is required. From this directory, create an isolated virtual environment if desired and run:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The verifier uses a temporary copy to preserve the archived evidence. It repeats the rational and numerical LP checks, verifies the original multistage certificates with exact arithmetic, and compares invariant fields with the archived results. LP floating residuals can vary by platform. To rerun the more expensive multistage numerical optimization, use `python reproduce.py --refit` (a fresh solver can produce a different feasible certificate). Original scripts write result files in their directory; run them in a working copy. See [paper.md](paper.md) for all formulas and scripts.

## Research files

| File | Purpose |
|---|---|
| [paper.md](paper.md) | Complete mathematical argument, assumptions, examples and related work |
| [barrier.py](barrier.py) | Exact witness construction and LP cross-checks |
| [exact_policy.py](exact_policy.py) | Exact rational polygon optimizer |
| [robust.py](robust.py) | Multistage LP and rational Bellman bounds |
| [verify_archived.py](verify_archived.py) | Exact check of archived feasible-policy certificates |
| [metr_points.csv](metr_points.csv) | Selected public METR values, separately attributed |
| [barrier_results.json](barrier_results.json), [exact_policy_results.json](exact_policy_results.json), [results.json](results.json) | Original recorded outputs and certificates |
| [archive_verification.json](archive_verification.json) | Publication-time archived certificate verification |
| [AGENTS.md](AGENTS.md), [llms.txt](llms.txt), [CITATION.cff](CITATION.cff), [codemeta.json](codemeta.json) | Agent-readable and scholarly attribution |

## Citation and scope

Easterbrook, Z. (2026). *The Adaptive Compute Barrier: A Sharp Minimax Boundary for Score-Aware AI Inference Budgets* (v0.1.0) [Preliminary AI-assisted research note and software]. GitHub. https://github.com/zakkeasterbrook/PA5/tree/main/research/adaptive-compute-barrier

Please cite the actual evaluated commit. This is a distinct companion study housed in the PA5 repository, not an extension of the original PA5 attention proof or part of the earlier PA5 v0.1.0 release. No DOI has been assigned to this study. Original project materials are offered under the repository's Apache-2.0 terms to the extent applicable rights are held; third-party data and references retain their own terms. Academic citation is encouraged. See [DATA_SOURCES.md](DATA_SOURCES.md), [NOTICE](NOTICE) and [the proof's limitations](paper.md#8-related-work-and-novelty-limits). Independent scrutiny, counterexamples and comparisons with prior art are welcome.
