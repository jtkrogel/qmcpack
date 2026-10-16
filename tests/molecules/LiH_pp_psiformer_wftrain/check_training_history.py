#!/usr/bin/env python3
"""Check that scalar statistical blocks and optimizer history have distinct counts."""
import csv
import math
from pathlib import Path
import sys

base = Path(sys.argv[1])
root = base.with_suffix(".s000")
with Path(str(root) + ".training_info.csv").open() as stream:
    rows = list(csv.DictReader(stream))
if Path(str(root) + ".scalar.dat").exists() or Path(str(root) + ".stat.h5").exists():
    raise SystemExit("Unexpected top-level scalar/stat output")
scalar_files = sorted(Path(str(root) + ".training").glob("*.scalar.dat"))
if len(rows) != 2 or len(scalar_files) != 2:
    raise SystemExit("Expected two optimizer attempts and four statistical blocks")
for index, row in enumerate(rows):
    expected = Path(str(root) + ".training") / (root.name + f".i{index:07d}.scalar.dat")
    if scalar_files[index] != expected:
        raise SystemExit("Incorrect iteration scalar filename")
    scalar_rows = [line for line in expected.read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
    stat_path = expected.with_name(expected.name.replace(".scalar.dat", ".stat.h5"))
    if len(scalar_rows) != 2 or stat_path.exists():
        raise SystemExit("Expected two scalar blocks per iteration and no stat HDF5")
    if int(row["iteration"]) != index or int(row["sample_version"]) != index:
        raise SystemExit("Iteration or sampled parameter version mismatch")
    if int(row["samples"]) != 8 or int(row["parameter_version"]) != index + 1:
        raise SystemExit("Pooled sample count or published version mismatch")
    for field in ["mean_energy", "variance", "gradient_norm", "sampling_seconds", "total_seconds"]:
        if not math.isfinite(float(row[field])):
            raise SystemExit("Non-finite iteration diagnostic: " + field)
    if float(row["total_seconds"]) < float(row["sampling_seconds"]):
        raise SystemExit("Full iteration time excludes its sampling time")
print("Iteration history and statistical scalar blocks retain independent, version-labelled semantics")
