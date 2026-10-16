#!/usr/bin/env python3
"""Small CPU learned-phase first-order/restart gates, not convergence benchmarks."""
import argparse
import csv
import hashlib
import math
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def text(dataset):
    """Decode the existing HDF string representation without a checkpoint-format change."""
    value = dataset[()]
    if isinstance(value, np.ndarray):
        value = value.item() if value.size == 1 else value.tobytes()
    return value.decode().rstrip("\x00") if isinstance(value, bytes) else str(value)


def checkpoint(directory):
    """Read only the root envelope selected by the atomically published manifest."""
    with h5py.File(directory / "checkpoint.manifest.h5", "r") as archive:
        filename = text(archive["wftrain_distributed_manifest/root_envelope_filename"])
    with h5py.File(directory / filename, "r") as archive:
        root = archive["wftrain_checkpoint"]
        return (root["model/values"][()].copy(),
                int(np.asarray(root["coordinator/completed_iterations"][()]).item()))


def input_xml(source, directory, optimizer, iterations, restart=None, cap=2147483648, wrong_twist=False):
    """Generate actual short XML by modifying only the existing fixed-position CPU fixture's driver."""
    root = ET.parse(source).getroot()
    root.find("project").set("id", "complex_cpu_training")
    for old in root.findall("qmc"):
        root.remove(old)
    if wrong_twist:
        root.find(".//psiformer").set("twist", "0.19 -0.11 0.07")
    qmc = ET.SubElement(root, "qmc", method="wf_train", move="pbyp", checkpoint="1")
    training = ET.SubElement(qmc, "training", objective="energy", optimizer=optimizer)
    controls = {"iterations": iterations, "learning_rate": "1e-7", "parameter_chunk_size": 4096,
                "maximum_retained_bytes": cap, "checkpoint_path": "checkpoint.manifest.h5"}
    if restart is not None:
        controls["restart_path"] = str(restart.resolve())
    for name, value in controls.items():
        ET.SubElement(training, "parameter", name=name).text = str(value)
    for name, value in {"walkers_per_rank": 1, "crowds": 1, "blocks": 2 if optimizer == "sgd" else 1, "steps": 1,
                        "warmupsteps": 0, "substeps": 1, "timestep": "1e-8", "usedrift": "no"}.items():
        ET.SubElement(qmc, "parameter", name=name).text = str(value)
    directory.mkdir()
    destination = directory / "input.xml"
    ET.ElementTree(root).write(destination, encoding="unicode", xml_declaration=True)
    return destination


def run(binary, source, expected=None):
    """Preserve each run and require an early diagnostic for negative admissions."""
    result = subprocess.run([str(binary), source.name], cwd=source.parent, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    (source.parent / "qmcpack.log").write_text(result.stdout)
    if expected is not None:
        if result.returncode == 0 or expected not in result.stdout:
            raise RuntimeError(f"Expected {expected!r}:\n{result.stdout[-6000:]}")
        if list(source.parent.glob("*.training.csv")):
            raise RuntimeError("Admission failure occurred after training history creation")
        return
    if result.returncode:
        raise RuntimeError(f"Training failed ({result.returncode}):\n{result.stdout[-6000:]}")
    histories = list(source.parent.glob("complex_cpu_training*.training.csv"))
    if len(histories) != 1:
        raise RuntimeError("Expected one actual series-qualified training history")
    with histories[0].open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["committed"] not in ("1", "true") or
                       not all(math.isfinite(float(row[name])) for name in
                               ("mean_energy", "variance", "gradient_norm")) for row in rows):
        raise RuntimeError("Missing finite, committed actual training history")


def main():
    """Gate compact Gamma Adam/twisted SGD, imaginary-head publication and exact continuation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="training_", dir=args.output.resolve()))
    binary, source = args.binary.resolve(), args.input.resolve()
    first_input = input_xml(source, work / "first", args.optimizer, 1)
    run(binary, first_input)
    first, iterations = checkpoint(first_input.parent)
    if iterations != 1 or not np.isfinite(first).all():
        raise RuntimeError("First update checkpoint is not finite and complete")
    manifest = first_input.parent / "checkpoint.manifest.h5"
    saved_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if args.optimizer == "adam":
        match = re.search(r"prospective named numeric bytes plus allowance=(\d+)",
                          (first_input.parent / "qmcpack.log").read_text())
        if match is None:
            raise RuntimeError("Missing preallocation admission diagnostic")
        exact_boundary = int(match.group(1))
        bad = input_xml(source, work / "wrong_twist", args.optimizer, 2, restart=manifest, wrong_twist=True)
        # Restore validates stage identity before opening the update transaction; history itself is initialized earlier.
        result = subprocess.run([str(binary), bad.name], cwd=bad.parent, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        (bad.parent / "qmcpack.log").write_text(result.stdout)
        if result.returncode == 0 or "stage identity or configuration mismatch" not in result.stdout:
            raise RuntimeError(f"Wrong-twist checkpoint was not rejected:\n{result.stdout[-6000:]}")
        if saved_hash != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise RuntimeError("Failed restore modified the source manifest")
        if (bad.parent / "checkpoint.manifest.h5").exists():
            raise RuntimeError("Failed restore published a new checkpoint")
        for name, optimizer, cap, diagnostic in (
                ("budget", "adam", exact_boundary - 1, "retained training storage exceeds maximum_retained_bytes"),
                ("optimizer", "rmsprop", 2147483648, "paired training requires")):
            run(binary, input_xml(source, work / name, optimizer, 1, cap=cap), diagnostic)
    resumed = input_xml(source, work / "resumed", args.optimizer, 2, restart=manifest)
    run(binary, resumed)
    second, iterations = checkpoint(resumed.parent)
    if iterations != 2 or not np.isfinite(second).all():
        raise RuntimeError("Continuation did not publish its second complete update")
    # FROZEN v1 FIXTURE OFFSETS, not archived block metadata: D16,N4,F256,Na2,4layers.
    # Initialization.cpp's canonical cusp2/envelope4*64*2 precedes up/down w,w_imag.
    expected_size = 2 + 4 * 64 * 2 + 4 * 256 * 64 + 15 * 256 + 4 * (6 * 256 * 256 + 2 * 256)
    if first.size != expected_size or second.size != expected_size:
        raise RuntimeError("Frozen v1 fixture layout changed; imaginary-head offsets need explicit requalification")
    head, prefix = 256 * 16 * 4, 2 + 4 * (16 * 4) * 2
    imaginary = np.r_[prefix + head:prefix + 2 * head, prefix + 3 * head:prefix + 4 * head]
    if second.size <= imaginary[-1] or not np.any(second[imaginary] != first[imaginary]):
        raise RuntimeError("No actual canonical imaginary-head parameter changed")
    if args.optimizer == "adam":
        continuous = input_xml(source, work / "continuous", args.optimizer, 2)
        run(binary, continuous)
        reference, iterations = checkpoint(continuous.parent)
        if iterations != 2 or not np.array_equal(reference, second):
            raise RuntimeError("Restarted canonical parameters differ from uninterrupted CPU Adam")
    print(f"PASS {args.optimizer}: complete CPU updates, nonzero imaginary-head change, checkpoint continuation")
    print(f"Evidence: {work}")


if __name__ == "__main__":
    main()
