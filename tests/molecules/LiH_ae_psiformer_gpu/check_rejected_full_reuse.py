#!/usr/bin/env python3
"""Check rejection-specific FULL reuse without adding production instrumentation.

Derive a high-tau, real/open AE training case from the unchanged low-tau VMC
fixture. Exact restart is within one backend. An optional pre42 executable
provides exact physical-state parity and the existing fine-timer count oracle.
This is not a throughput, timing, memory, or statistical sampling benchmark.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter(parent, name, value):
    ET.SubElement(parent, "parameter", name=name).text = str(value)


def render(template, destination, project, iterations, tau, restart=None):
    tree = ET.parse(template)
    root = tree.getroot()
    root.find("project").set("id", project)
    root.find("wavefunction/psiformer").attrib.update(
        gpu="cuda", optimize="yes", optimize_scope="indices", optimize_indices="0")
    root.remove(root.find("walkerlogs"))
    root.remove(root.find("qmc"))
    qmc = ET.SubElement(root, "qmc", method="wf_train", move="alle", checkpoint="-1")
    training = ET.SubElement(qmc, "training", objective="energy", optimizer="adam")
    for name, value in dict(iterations=iterations, warmupsteps=0, learning_rate="1e-10",
                            parameter_chunk_size=4096, checkpoint_interval=1000).items():
        parameter(training, name, value)
    if restart:
        parameter(training, "restart_path", restart)
    for name, value in dict(walkers_per_rank=1, crowds=1, blocks=2, steps=1,
                            warmupsteps=0, substeps=1, timestep=tau, usedrift="no",
                            blocks_between_recompute=100).items():
        parameter(qmc, name, value)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def envelopes(latest):
    import h5py
    def text(group, key):
        value = group[key][()]
        if not isinstance(value, (str, bytes)):
            value = value.flat[0]
        return value.decode() if isinstance(value, bytes) else value
    with h5py.File(latest) as handle:
        manifest = latest.parent / text(handle, "wftrain_latest_checkpoint/manifest_relative_path")
    with h5py.File(manifest) as handle:
        root = manifest.parent / text(handle, "wftrain_distributed_manifest/root_envelope_filename")
        shard = manifest.parent / (text(handle, "wftrain_distributed_manifest/sampler_shard_prefix") + ".rank_000000.h5")
    if not root.is_file() or not shard.is_file():
        raise AssertionError("Missing committed ROOT or rank-zero sampler envelope")
    return root, shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("--baseline-app", type=Path, help="Frozen pre42 CUDA app; enables measured skip proof")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tau", type=float, default=64)
    parser.add_argument("--device-bytes", type=int, default=3 * 1024**3)
    parser.add_argument("--h5diff", default=shutil.which("h5diff"))
    args = parser.parse_args()
    if not math.isfinite(args.tau) or args.tau <= 0 or args.device_bytes <= 0 or not args.h5diff:
        parser.error("positive finite tau, positive device cap and h5diff are required")
    args.app = args.app.resolve()
    args.source = args.source.resolve()
    template = args.source / "tests/molecules/LiH_ae_psiformer_gpu/input.xml"
    original_hash = sha(template)
    if args.output:
        args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    else:
        args.output = Path(tempfile.mkdtemp(prefix="psiformer_rejected_full_"))
    print("Rejection-specific real LiH AE proof:", args.output, flush=True)
    deadline = time.monotonic() + 300
    def execute(command, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Native rejection-reuse suite exceeded 300 seconds")
        subprocess.run(command, timeout=min(180, remaining), check=True, **kwargs)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES=str(args.device_bytes),
                       PSIFORMER_GPU_METROPOLIS_BYTES=str(16 * 1024**2))
    environment.pop("PSIFORMER_GPU_METROPOLIS_DRIFT_BYTES", None)
    def run(app, directory, project, iterations, restart=None):
        directory.mkdir(exist_ok=True)
        xml = directory / f"{project}.input.xml"
        render(template, xml, project, iterations, args.tau, restart)
        log_path = directory / f"{project}.run.log"
        with log_path.open("w") as output:
            execute([str(app), "--enable-timers=fine", xml.name], cwd=directory,
                    env=environment, stdout=output, stderr=subprocess.STDOUT)
        log = log_path.read_text()
        windows = re.findall(r"device sampling completed:[^\n]*uniform_draws=(\d+), accepts=(\d+)", log)
        expected = 2 * (iterations - (1 if restart else 0))
        if len(windows) != expected or any(draws != "1" or accepts != "0" for draws, accepts in windows):
            raise AssertionError("Expected only regular eligible all-rejected sampling windows")
        with (directory / f"{project}.s000.training_info.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if any(row["committed"] != "1" for row in rows) or int(rows[-1]["parameter_version"]) != iterations:
            raise AssertionError("Native training update/version failure")
        for row in rows:
            if not all(math.isfinite(float(row[key])) for key in ("mean_energy", "variance", "gradient_norm")):
                raise AssertionError("Nonfinite native physical history")
        info = ET.parse(directory / f"{project}.info.xml")
        calls = sum(int(timer.findtext("calls")) for timer in info.findall(".//timer")
                    if timer.findtext("name") == "PsiFormer::batch_VGL")
        return directory / f"{project}.s000.last_checkpoint.h5", rows[-1], calls, len(windows)
    def compare(original, restored):
        first, second = envelopes(original[0]), envelopes(restored[0])
        if len(first) != len(second) or len(first) != 2:
            raise AssertionError("Checkpoint envelope extent mismatch")
        for left, right in zip(first, second):
            execute([args.h5diff, str(left), str(right)])
        for key in ("iteration", "sample_version", "parameter_version", "samples", "mean_energy",
                    "variance", "gradient_norm", "selected_alpha", "committed"):
            if original[1][key] != restored[1][key]:
                raise AssertionError("Exact saved physical history mismatch: " + key)
    continuous = run(args.app, args.output / "continuous", "reuse_continuous", 2)
    split = args.output / "split"
    first = run(args.app, split, "reuse_checkpoint", 1)
    resumed = run(args.app, split, "reuse_restart", 2, first[0].name)
    compare(continuous, resumed)
    baseline_calls = None
    if args.baseline_app:
        baseline = run(args.baseline_app.resolve(), args.output / "baseline", "reuse_continuous", 2)
        compare(continuous, baseline)
        baseline_calls = baseline[2]
        if baseline_calls - continuous[2] != continuous[3]:
            raise AssertionError("Existing FULL timer does not prove one skipped refresh per rejected window")
    if sha(template) != original_hash:
        raise AssertionError("Protected low-tau AE fixture changed")
    report = dict(app_sha256=sha(args.app), protected_template_sha256=original_hash, physics="real LiH AE",
                  tau=args.tau, exact_restart=True, rejected_windows=continuous[3], full_calls=continuous[2],
                  baseline_full_calls=baseline_calls, skip_measured=bool(args.baseline_app),
                  baseline_app_sha256=sha(args.baseline_app.resolve()) if args.baseline_app else None,
                  device_admission_cap=args.device_bytes, benchmark=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS:", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
