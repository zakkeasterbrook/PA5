"""Audit committed aggregate results against original per-trial CSVs."""
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent


def load_csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    trials = load_csv("trials.csv")
    stress = load_csv("stress_trials.csv")
    summary = load_csv("summary.csv")
    assert len(trials) == 288 and sum(int(r["correct"]) for r in trials) == 288
    dual = [r for r in trials if r["mode"] == "dual"]
    assert len(dual) == 96 and sum(int(r["accepted"]) for r in dual) == 62
    assert len(stress) == 4000 and sum(int(r["correct"]) for r in stress) == 4000
    counts = json.loads((ROOT / "stress_results.json").read_text())
    assert counts["accepted"] == sum(int(r["accepted"]) for r in stress) == 1895
    assert counts["disagreements"] == 0
    assert len(summary) == 72
    groups = ("n", "kind", "mode")
    assert len({tuple(r[k] for k in groups) for r in summary}) == 72
    for s in summary:
        rows = [r for r in trials if all(r[k] == s[k] for k in groups)]
        assert len(rows) == 4
        for short, field in (("read", "read_fraction"), ("keys", "key_fraction"),
                             ("values", "value_fraction"), ("accept", "accepted"),
                             ("ms", "ms")):
            agg = median if short == "ms" else mean
            expected = agg(float(r[field]) for r in rows)
            assert math.isclose(float(s[short]), expected, rel_tol=1e-10, abs_tol=1e-10), (s, short)
    print("PASS: 288 benchmark evaluations; 96 dual fixtures; 62 accepted.")
    print("PASS: 4,000 stress evaluations; 1,895 accepted; zero disagreements.")
    print("PASS: all 72 aggregate rows reproduce from original trial data.")
    print("Audit verifies recorded data and arithmetic, not independent experiment execution.")


if __name__ == "__main__":
    main()
