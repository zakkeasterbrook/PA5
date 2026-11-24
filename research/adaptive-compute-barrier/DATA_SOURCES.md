# Data provenance and reproduction scope

**Original data credit: METR.** Megan Kinniment et al., *AI R&D-relevant model capabilities are improving rapidly and are getting harder to measure*, November 20, 2026.

- Report: https://metr.org/blog/2026-11-20-q4-2026-capability-update/
- Published chart data: https://metr.org/assets/images/q4-2026-capability-update/fe5f1999eeec0af5.json
- Relevant report sections: Appendix I, Figure 58 (inference cost); Appendix N (Earthborne Rangers).

`metr_points.csv` contains 221 selected factual chart values transcribed in the earlier analysis (not an original experimental dataset). This study uses only 98 Earthborne Rangers rows: 48 GPT-6 Astra and 50 GPT-5.6 Sol checkpoints. The remaining 123 AZsPCs rows are retained for provenance but excluded from this study. All source model labels are preserved. The report and JSON were rechecked on November 24, 2026, and selected entries spot-checked against the file (including the first Astra checkpoint at $30.0437). A complete fresh source download was unavailable in the execution environment, so no full independent upstream row-by-row verification is claimed. The scripts use the selected rounded displayed values as exact rationals; they do not infer or reconstruct individual trajectories. The CSV is a selected transcription, not the full published curves.

The math assumes the means describe a common population of retained nondecreasing, observable, verifiable scores on the 21-objective grid at deterministic budget checkpoints. These assumptions require run-level verification for any deployment or empirical inference; source scoring, aggregation, interpolation, incomplete trajectories, verifier overhead, sampling uncertainty and billing may change conclusions. Costs are modeled checkpoints, not actual invoices. No new model runs were performed.

Apache-2.0 applies to the original project materials to the extent the author has rights; it is not a claim that METR's data, report, or any third-party work is licensed by this project. Consult the original source for applicable terms and cite METR when using its data. The authorship of this mathematical analysis does not imply endorsement or affiliation with METR or any AI laboratory.
