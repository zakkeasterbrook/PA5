# PA5 v0.1.0 — preliminary research software

Prepared release notes; the existence of this file is not a claim that a GitHub Release or Zenodo record has already been published.

**Author and project owner:** Zakk Easterbrook  
**Initial research date:** November 24, 2026  
**License:** Apache-2.0  
**Research status:** AI-assisted, not peer-reviewed.

## Included

- Research white paper, proof assumptions, and related work.
- Original reference algorithm and a separately documented, optional range-screened numerical wrapper.
- Original benchmark, exact-rational, and adversarial trial records.
- Published post-publication numerical counterexample and reproduced numerical audit.
- One-command reproduction suite and SHA-256 manifest for 16 archived files.
- Citation File Format, BibTeX, CodeMeta, Zenodo and AI-agent-readable metadata.

## Reproduce

```sh
python -m pip install -r requirements.txt
python reproduce.py
# Optional full benchmark:
python reproduce.py --full
```

The local quick reproduction check passed all archived hash checks, exact-rational checks, 4,000 original stress evaluations, 3,600 numerical regression evaluations and 72 non-timing quick-benchmark records. Earlier full reproduction matched all 288 non-timing benchmark records; see [AUDIT.md](AUDIT.md). The new full reproduction wrapper is intended for a longer-running environment; the latest local invocation exceeded its execution time limit. Timing varies by hardware and is not asserted.

## Scientific claims and limitations

At 524,288 synthetic attention entries with concentrated attention, the original experiment saved 94.5% of modeled logical cache reads and measured a 4.92x CPU speed ratio, excluding cache construction. Random and tied cases were slower. The theorem applies to exact real arithmetic for a restricted bias-free linear output readout. The original numerical implementation has a documented incorrect-acceptance counterexample. The guarded wrapper mitigates that example and passed the recorded additional tests but is not a rigorous interval implementation. No general LLM inference speedup, GPU improvement, storage reduction, priority, or universal finite-precision correctness is claimed.

## Citation

Easterbrook, Z. (2026). *Predictive Atom 5 (PA5): Decision-Certified Key-Value Cache Retrieval for Efficient AI Inference* (v0.1.0) [Research software and preliminary research note]. https://github.com/zakkeasterbrook/PA5.

See [CITATION.cff](CITATION.cff), [CITATION.bib](CITATION.bib), [LICENSE](LICENSE), and [NOTICE](NOTICE). Include the exact commit when reporting reproduction results. A DOI is pending archive publication; none is currently asserted.
