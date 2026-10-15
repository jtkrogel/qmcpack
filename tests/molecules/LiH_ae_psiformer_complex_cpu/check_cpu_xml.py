#!/usr/bin/env python3
"""Run one explicit-position CPU XML smoke; this is not an energy-convergence test."""
import argparse
import math
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    """Require production qmcpack success and finite scalar observations for two blocks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-selected-system-rejection", action="store_true")
    args = parser.parse_args()
    binary, source, output = args.binary.resolve(), args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run_", dir=output))
    destination = work / "input.xml"
    shutil.copyfile(source, destination)
    run = subprocess.run([str(binary), destination.name], cwd=work, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    (work / "qmcpack.log").write_text(run.stdout)
    if args.expect_selected_system_rejection:
        if run.returncode == 0 or "Complex PsiFormer selected-system admission" not in run.stdout:
            raise RuntimeError(f"Expected actual selected-system rejection:\n{run.stdout[-6000:]}")
        if list(work.glob("*.scalar.dat")):
            raise RuntimeError("Selected-system rejection occurred after sampling output was created")
        print(f"PASS {source.name}: selected-system rejected before sampling; evidence {work}")
        return
    if run.returncode:
        raise RuntimeError(f"qmcpack failed ({run.returncode}):\n{run.stdout[-6000:]}")
    files = list(work.glob("complex_cpu.s000.scalar.dat"))
    if len(files) != 1:
        raise RuntimeError("Missing production scalar output")
    rows = [line.split() for line in files[0].read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != 2 or not all(math.isfinite(float(value)) for row in rows for value in row):
        raise RuntimeError("Expected two finite initialized-model blocks")
    print(f"PASS {source.name}: two finite CPU blocks; no convergence or cross-trajectory claim")
    print(f"Evidence: {work}")


if __name__ == "__main__":
    main()
