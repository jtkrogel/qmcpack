#!/usr/bin/env python3
"""One internally initialized Gamma paired-BFD pilot and exact initialized restart."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def output_directory(parent):
    """Keep every invocation's evidence; never remove a prior run."""
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "source", "output-parent"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--reference", action="store_true", help="Use the qualified W2 fixed-grid reference profile")
    args = parser.parse_args()
    assert (
        subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
        == args.source_commit
    ), "Source changed: reconfigure CMake"
    root = output_directory(args.output_parent)
    helpers = Path(__file__).resolve().parent
    deadline = time.monotonic() + 190
    active = None

    def cancel():
        # Each proven helper catches TERM and reaps its own separate native PG.
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()

    def interrupted(signum, frame):
        cancel()
        raise SystemExit("paired ECP validation interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def run(script, *arguments):
        nonlocal active
        active = subprocess.Popen(
            [sys.executable, str(helpers / script), *map(str, arguments)],
            start_new_session=True,
        )
        try:
            assert active.wait(timeout=max(0.001, deadline - time.monotonic())) == 0, (
                script
            )
        except subprocess.TimeoutExpired:
            cancel()
            raise AssertionError("paired ECP validation exceeded190s aggregate")
        finally:
            cancel()
            active = None

    run(
        "prepare_paired_ecp_input.py",
        "--source",
        args.source,
        "--output",
        root / "prepared",
    )
    run(
        "check_paired_ecp_pilot.py",
        "--binary",
        args.binary,
        "--input",
        root / "prepared/input.xml",
        "--output",
        root / "pilot",
        "--source-commit",
        args.source_commit,
        *(["--reference"] if args.reference else []),
    )
    run(
        "check_paired_ecp_restart.py",
        "--binary",
        args.binary,
        "--source",
        args.source,
        "--pilot-report",
        root / "pilot/report.json",
        "--output",
        root / "restart",
        "--pilot-helper",
        helpers / "check_paired_ecp_pilot.py",
    )
    report = json.loads((root / "restart/report.json").read_text())
    assert report["status"] == "PASS_EXACT_INITIALIZED_RESTART"
    marker = "PASS_PAIRED_PP_REFERENCE_EXACT_RESTART" if args.reference else "PASS_PAIRED_PP_EXACT_RESTART"
    print(marker + " evidence=" + str(root), flush=True)


if __name__ == "__main__":
    main()
