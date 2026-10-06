"""Combine per-rank-count benchmark JSON files into one machine-generated
Markdown summary spanning every rank count that was run.

``benchmark.py`` writes one ``benchmark_results_n<ranks>.json`` per
invocation (one process group's own view). ``test.sh`` already runs
``benchmark.py`` at ``n=4,8,16,$SLURM_NTASKS``, so after a full test.sh
pass there are several such files sitting side by side; this script reads
all of them and prints (and writes) one table with every (method, ranks)
row together, sorted by method then by rank count, matching the exact
column layout the project's benchmarking spec calls for.

Run with (after a test.sh pass, or after several manual
``mpirun -n <N> python benchmark.py`` runs at different N):

    python summarize_benchmarks.py [glob]

``glob`` defaults to ``benchmark_results_n*.json`` in the current
directory.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "pattern",
    nargs="?",
    default="benchmark_results_n*.json",
    help="glob pattern for per-rank-count benchmark JSON files",
)
parser.add_argument(
    "--out",
    default="benchmark_summary.md",
    help="output Markdown file",
)
args = parser.parse_args()

paths = sorted(glob.glob(args.pattern))
if not paths:
    print(f"no files matched {args.pattern!r}; run benchmark.py first", file=sys.stderr)
    raise SystemExit(1)

rows: list[dict] = []
seen_ranks: set[int] = set()
for path in paths:
    with open(path) as f:
        payload = json.load(f)
    ranks = payload["ranks"]
    if ranks in seen_ranks:
        # A rerun at the same rank count overwrote/duplicated a file for
        # that N; last one wins rather than silently duplicating rows.
        rows = [r for r in rows if r["ranks"] != ranks]
    seen_ranks.add(ranks)
    for r in payload["results"]:
        rows.append(r)

# Stable, readable ordering: group by method, then by ascending rank
# count within a method, mirroring how a reader compares scaling for one
# operation at a time rather than one rank count at a time.
rows.sort(key=lambda r: (r["op"], r["ranks"]))

lines = [
    "| Method | Ranks | Native Xarray | MPI Xarray | Speedup | Accuracy | Dtype |",
    "| --- | ---: | ---: | ---: | ---: | --- | --- |",
]
n_slower = 0
for r in rows:
    if r.get("error") is not None:
        lines.append(
            f"| `{r['op']}` | {r['ranks']} | FAILED | FAILED | n/a | "
            f"{r['accuracy']} | {r['dtype']} |  <!-- {r['error']} -->"
        )
        continue
    nat = f"{r['native_s']:.4f} s" if r.get("native_s") is not None else "n/a"
    mpi_s = f"{r['mpi_s']:.4f} s" if r.get("mpi_s") is not None else "n/a"
    speedup = r.get("speedup")
    if speedup is not None:
        sp = f"{speedup:.2f}x"
        if speedup < 1.0:
            sp += " SLOWER"
            n_slower += 1
    elif r.get("no_native_counterpart"):
        sp = "n/a (no native counterpart)"
    else:
        sp = "n/a"
    lines.append(
        f"| `{r['op']}` | {r['ranks']} | {nat} | {mpi_s} | {sp} | "
        f"{r['accuracy']} | {r['dtype']} |"
    )

summary = "\n".join(lines)
header = (
    f"Combined from {len(paths)} rank-count run(s): "
    f"{sorted(seen_ranks)}. {len(rows)} (method, ranks) rows, "
    f"{n_slower} slower-than-native at their run's problem size.\n\n"
)
print(header + summary)
with open(args.out, "w") as f:
    f.write(header + summary + "\n")
