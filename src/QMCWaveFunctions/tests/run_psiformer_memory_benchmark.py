#!/usr/bin/env python3
"""Run each PsiFormer memory scenario in a fresh process.

The fixture-generation process exits before any measurement process starts, so
its HDF5 buffers cannot pollute scenario peak RSS.  Individual JSON documents
remain useful on their own; index.json records the exact commands and embeds
their reports for convenient comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_SCENARIOS = (
    "model_load",
    "clone_population",
    "value",
    "full_vgl",
    "active_gradient",
    "crowd_value",
    "crowd_full_vgl",
    "crowd_active_gradient",
    "score",
    "score_and_kinetic",
)

NATIVE_SCENARIOS = {
    "value",
    "full_vgl",
    "active_gradient",
    "score",
    "score_and_kinetic",
}


def parse_args() -> argparse.Namespace:
    """Parse reproducibility and workload controls for the process launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="benchmark_psiformer_memory executable")
    parser.add_argument("output_dir", type=Path, help="directory for fixture and JSON reports")
    parser.add_argument(
        "--native-executable",
        type=Path,
        help="native-workspace executable (default: sibling benchmark_psiformer_native_memory)",
    )
    parser.add_argument("--walkers", type=int, default=4, help="component/crowd population (default: 4)")
    parser.add_argument("--warm-calls", type=int, default=1, help="calls after first touch (default: 1)")
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="comma-separated scenario subset",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="OMP and BLAS threads recorded in each report (default: 1)",
    )
    return parser.parse_args()


def checked_scenarios(specification: str) -> list[str]:
    """Validate scenario names before creating any output files."""
    scenarios = [name.strip() for name in specification.split(",") if name.strip()]
    unknown = sorted(set(scenarios) - set(DEFAULT_SCENARIOS))
    if not scenarios or unknown:
        raise ValueError(f"invalid scenario list; unknown={unknown}, accepted={DEFAULT_SCENARIOS}")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("scenario list contains duplicates")
    return scenarios


def thread_environment(threads: int) -> dict[str, str]:
    """Pin the common host threading runtimes to one explicit subscription."""
    environment = dict(os.environ)
    count = str(threads)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = count
    environment.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
    environment.setdefault("OMP_PROC_BIND", "spread")
    environment.setdefault("OMP_PLACES", "cores")
    return environment


def run_command(command: list[str], environment: dict[str, str]) -> None:
    """Echo and execute one benchmark command with failure propagation."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, env=environment, check=True)


def main() -> int:
    """Generate one fixture, launch fresh processes, and write an index."""
    arguments = parse_args()
    if arguments.walkers <= 0 or arguments.warm_calls < 0 or arguments.threads <= 0:
        raise ValueError("walkers and threads must be positive; warm calls must be nonnegative")

    executable = arguments.executable.resolve(strict=True)
    native_executable = (
        arguments.native_executable
        if arguments.native_executable is not None
        else executable.with_name("benchmark_psiformer_native_memory")
    ).resolve(strict=True)
    scenarios = checked_scenarios(arguments.scenarios)
    output_dir = arguments.output_dir.resolve()
    fixture_dir = output_dir / "fixture_lih"
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = thread_environment(arguments.threads)

    fixture_command = [str(executable), "--generate-fixture", "lih", str(fixture_dir)]
    run_command(fixture_command, environment)
    parameter_file = fixture_dir / "parameters.h5"
    configuration_file = fixture_dir / "configuration.h5"

    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        report_file = output_dir / f"{scenario}.json"
        scenario_executable = native_executable if scenario in NATIVE_SCENARIOS else executable
        command = [
            str(scenario_executable),
            "--scenario",
            scenario,
            "--parameters",
            str(parameter_file),
            "--configuration",
            str(configuration_file),
            "--walkers",
            str(arguments.walkers),
            "--warm-calls",
            str(arguments.warm_calls),
            "--output",
            str(report_file),
        ]
        run_command(command, environment)
        with report_file.open(encoding="utf-8") as stream:
            report = json.load(stream)
        records.append({"scenario": scenario, "command": command, "report": report})

    index = {
        "schema": "qmcpack.psiformer.memory_benchmark_index.v1",
        "fixture_command": fixture_command,
        "component_executable": str(executable),
        "native_executable": str(native_executable),
        "thread_environment": {
            name: environment[name]
            for name in (
                "OMP_NUM_THREADS",
                "OMP_MAX_ACTIVE_LEVELS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "records": records,
    }
    index_path = output_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as stream:
        json.dump(index, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {index_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
