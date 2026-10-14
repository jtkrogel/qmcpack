#!/usr/bin/env python3
"""Check that scalar statistical blocks and optimizer history have distinct counts."""
import csv
import math
from pathlib import Path
import sys

base = Path(sys.argv[1])
with base.with_suffix(".s000.training.csv").open() as stream:
    rows = list(csv.DictReader(stream))
scalar = base.with_suffix(".s000.scalar.dat").read_text().splitlines()
scalar_rows = [line for line in scalar if line.strip() and not line.lstrip().startswith("#")]
if len(rows) != 2 or len(scalar_rows) != 4:
    raise SystemExit("Expected two optimizer attempts and four statistical blocks")
for index, row in enumerate(rows):
    if int(row["iteration"]) != index + 1 or int(row["sample_version"]) != index:
        raise SystemExit("Iteration or sampled parameter version mismatch")
    if int(row["samples"]) != 8 or int(row["parameter_version"]) != index + 1:
        raise SystemExit("Pooled sample count or published version mismatch")
    for field in ["mean_energy", "variance", "gradient_norm", "sampling_seconds", "total_seconds"]:
        if not math.isfinite(float(row[field])):
            raise SystemExit("Non-finite iteration diagnostic: " + field)
    if float(row["total_seconds"]) < float(row["sampling_seconds"]):
        raise SystemExit("Full iteration time excludes its sampling time")
print("Iteration history and statistical scalar blocks retain independent, version-labelled semantics")
