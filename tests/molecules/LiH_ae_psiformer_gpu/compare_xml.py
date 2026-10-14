#!/usr/bin/env python3
"""Compare matched native-initialized CPU/CUDA VMC without external data."""
import argparse
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def scalar_data(filename):
    """Read named scalar columns, preserving non-timing statistical observables."""
    lines = filename.read_text().splitlines()
    header = lines[0].lstrip("#").split()
    selected = [i for i, name in enumerate(header) if name not in {"BlockCPU", "BlockTime", "Time"}]
    rows = [[float(token) for token in line.split()] for line in lines[1:] if line.strip()]
    if not rows or any(len(row) != len(header) for row in rows):
        raise RuntimeError("Scalar output is empty or structurally inconsistent")
    return header, [[row[i] for i in selected] for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--twist", help="Explicit fixed outer twist; requires a complex executable")
    parser.add_argument("--mixed", action="store_true", help="Qualify explicitly mixed CUDA against CPU FP64")
    args = parser.parse_args()
    results = {}
    for backend in ("cpu", "cuda"):
        directory = args.output / backend
        directory.mkdir(parents=True, exist_ok=True)
        tree = ET.parse(args.input)
        tree.getroot().find(".//wavefunction/psiformer").set("gpu", backend)
        if args.mixed:
            tree.getroot().find(".//wavefunction/psiformer").set(
                "precision", "fp32_dense_fp64_islands" if backend == "cuda" else "fp64_reference")
        if args.twist is not None:
            tree.getroot().find(".//wavefunction/psiformer").set("twist", args.twist)
        tree.write(directory / "input.xml", encoding="unicode", xml_declaration=True)
        with (directory / "console.log").open("w") as log:
            completed = subprocess.run([str(args.binary.resolve()), "input.xml"], cwd=directory,
                                       stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{backend} run failed; inspect {directory / 'console.log'}")
        console = (directory / "console.log").read_text()
        if backend == "cuda" and "numerical device tile completed: backend=cuda" not in console:
            raise RuntimeError("Strict CUDA run produced no completed-device-tile diagnostic")
        if backend == "cuda" and args.mixed:
            if "mixed inference completed: mixed_batches=1, whole_fp64_retries=0" not in console:
                raise RuntimeError("Explicit mixed run did not report actual mixed execution without fallback")
        results[backend] = scalar_data(directory / "check.s000.scalar.dat")
    if results["cpu"][0] != results["cuda"][0] or len(results["cpu"][1]) != len(results["cuda"][1]):
        raise RuntimeError("CPU/CUDA scalar schemas differ")
    maximum = 0.0
    tolerance = 5e-4 if args.mixed else 2e-6
    for cpu, cuda in zip(results["cpu"][1], results["cuda"][1]):
        for left, right in zip(cpu, cuda):
            if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
                raise RuntimeError(f"CPU/CUDA observable mismatch: {left} versus {right}")
            maximum = max(maximum, abs(left - right))
    print(f"PASS native XML CPU/CUDA inference; maximum observable difference {maximum:.12g}")


if __name__ == "__main__":
    main()
