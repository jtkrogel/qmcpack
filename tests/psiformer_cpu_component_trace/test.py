#!/usr/bin/env python3
"""Compile/run the standard-library-only helper and validate off/on/error modes."""
import csv
import math
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix="psiformer_trace_test_") as temp:
    temp = Path(temp)
    executable = temp / "helper"
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-Wextra",
                    "-Werror", "-pthread", "-I", str(root / "src"),
                    str(Path(__file__).with_name("helper.cpp")), "-o", str(executable)], check=True)
    env = os.environ.copy()
    env.pop("PSIFORMER_CPU_COMPONENT_TRACE", None)
    assert subprocess.check_output([executable], env=env, text=True) == "0\n0\n"
    assert not list(temp.glob("*.tsv"))
    env["PSIFORMER_CPU_COMPONENT_TRACE"] = str(temp / "trace")
    env["OMPI_COMM_WORLD_RANK"] = "3"
    assert subprocess.check_output([executable], env=env, text=True) == "4\n4\n"
    paths = sorted(temp.glob("trace.run*.rank3.thread*.tsv"))
    assert len(paths) == 4, paths
    threads = set()
    for path in paths:
        with path.open() as source:
            reader = csv.DictReader(source, delimiter="\t")
            assert reader.fieldnames == ["schema", "rank", "thread", "stage", "calls", "seconds"]
            rows = list(reader)
        assert len(rows) == 3
        counts = {row["stage"]: int(row["calls"]) for row in rows}
        assert counts == {"value.envelope.exclusive": 7, "value.head.exclusive": 7,
                          "value.determinant.exclusive": 1}, counts
        for row in rows:
            assert row["schema"] == "1" and row["rank"] == "3"
            seconds = float(row["seconds"])
            assert math.isfinite(seconds) and seconds >= 0
            threads.add(row["thread"])
    assert len(threads) == 4
    env["PSIFORMER_CPU_COMPONENT_TRACE"] = str(temp / "missing_directory" / "trace")
    assert subprocess.check_output([executable], env=env, text=True) == "0\n0\n"
print("PASS: disabled output, four surviving worker records, TSV schema, counts, times, cumulative flush, I/O failure")
