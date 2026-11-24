"""Reproduce PA5 in an isolated directory; never overwrite recorded evidence."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Repeat the 524,288-entry benchmark instead of the quick benchmark")
    parser.add_argument("--output", type=Path, help="Write the new reproduction report to this path")
    args = parser.parse_args()
    import numpy as np
    check(np.__version__ == "2.3.5", "Install the pinned NumPy version from requirements.txt")
    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    report = {"python": platform.python_version(), "numpy": np.__version__,
              "platform": platform.platform(), "benchmark": "full" if args.full else "quick",
              "checks": {}, "known_original_float64_counterexample": True}
    manifest = ROOT / "ARTIFACTS.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, filename = line.split(maxsplit=1)
        path = ROOT / filename
        check(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Archived artifact hash mismatch: {filename}")
    report["checks"]["archive_integrity"] = "pass"
    with tempfile.TemporaryDirectory(prefix="pa5-reproduce-") as temp:
        scratch = Path(temp)
        def run(script, *extra):
            print(f"Running {script} ...", flush=True)
            return subprocess.run([sys.executable, str(ROOT / script), *extra], cwd=scratch,
                                  env=env, check=True, text=True, stdout=subprocess.PIPE).stdout
        run("demo.py")
        run("audit_results.py")
        report["checks"]["demo_and_recorded_aggregates"] = "pass"
        exact = json.loads(run("exact_check.py"))
        check(exact == json.loads((ROOT / "exact_results.json").read_text()), "Exact arithmetic results differ")
        report["checks"]["exact_rational"] = exact
        stress = json.loads(run("stress.py"))
        check(stress == json.loads((ROOT / "stress_results.json").read_text()), "Stress summary differs")
        check(read_csv(scratch / "stress_trials.csv") == read_csv(ROOT / "stress_trials.csv"), "Stress trial records differ")
        report["checks"]["original_stress"] = stress
        run("audit_numerics.py")
        numerical = json.loads((scratch / "numerical_audit_results.json").read_text())
        check(numerical == json.loads((ROOT / "numerical_audit_results.json").read_text()), "Numerical audit results differ")
        report["checks"]["numerical_regression"] = numerical
        output = scratch / "benchmark.csv"
        run("benchmark.py", *([] if args.full else ["--quick"]), "--out", str(output))
        baseline = ROOT / ("trials.csv" if args.full else "quick.csv")
        def strip_timing(rows):
            return [{k: v for k, v in row.items() if k not in ("ms", "build_ms")} for row in rows]
        observed = read_csv(output)
        check(strip_timing(observed) == strip_timing(read_csv(baseline)), "Benchmark non-timing fields differ from the published records")
        report["checks"]["benchmark_rows_matched"] = len(observed)
    report["status"] = "pass"
    report["scope"] = "Internal reproduction of synthetic CPU experiments; not a universal numerical proof or real-model performance test."
    body = json.dumps(report, indent=2) + "\n"
    if args.output:
        target = args.output.resolve()
        check(not target.exists(), "Refusing to overwrite an existing output file")
        target.write_text(body, encoding="utf-8")
    print(body, end="")


if __name__ == "__main__":
    main()
