#!/usr/bin/env python3
"""Bounded real LiH BFD ordinary-locality training and exact-restart check.

Derive inputs from the existing index-scoped production-architecture fixture.
This is BFD, not ccECP or the old all-electron GPU fixture. CPU/GPU trajectory
comparison is tolerance-based; exact saved-state equality applies only to split
restart on the same executable/backend. Run only on a qualified frozen binary.
"""
import argparse
import copy
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

from compare_ecp_restart import envelopes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter(parent, name, value):
    node = parent.find(f"parameter[@name='{name}']")
    if node is None:
        node = ET.SubElement(parent, "parameter", name=name)
    node.text = str(value)


def remaining(args):
    seconds = min(args.timeout, args.deadline - time.monotonic()) if args.deadline else args.timeout
    if seconds <= 0:
        raise TimeoutError("Native ECP suite deadline exceeded")
    return seconds


def render(template, destination, project, backend, iterations, args, restart=None):
    tree = ET.parse(template)
    root = tree.getroot()
    root.find("project").set("id", project)
    root.find("wavefunction/psiformer").set("gpu", backend)
    if args.resident_ecp_profile:
        head = root.find("wavefunction/psiformer")
        head.set("optimize_scope", "all")
        head.attrib.pop("optimize_indices", None)
    qmc = root.find("qmc")
    training = qmc.find("training")
    parameter(training, "iterations", iterations)
    if restart:
        parameter(training, "restart_path", restart)
    parameter(qmc, "walkers_per_rank", args.walkers)
    parameter(qmc, "blocks", args.blocks)
    if args.resident_ecp_profile:
        qmc.set("move", "alle")
        parameter(qmc, "blocks_between_recompute", 100)
        parameter(training, "maximum_retained_bytes", 1024**3)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def history(directory, project):
    with (directory / f"{project}.s000.training_info.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["committed"] != "1" for row in rows):
        raise AssertionError("Native BFD training did not commit every reported update")
    for row in rows:
        for key in ("mean_energy", "variance", "gradient_norm"):
            if not math.isfinite(float(row[key])):
                raise AssertionError("Nonfinite native BFD history: " + key)
    return rows


def run(app, directory, project, backend, iterations, args, restart=None):
    template = args.source / "tests/molecules/LiH_pp_psiformer_wftrain/wftrain_ecp_checkpoint.xml"
    for element in ("Li", "H"):
        target = directory / f"{element}.BFD.xml"
        if not target.exists():
            target.symlink_to(args.source / f"tests/molecules/LiH_pp/{element}.BFD.xml")
    render(template, directory / "run.xml", project, backend, iterations, args, restart)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if backend == "cuda":
        environment["PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES"] = str(args.device_bytes)
    if args.resident_ecp_profile:
        environment = {key: value for key, value in environment.items() if not key.startswith("PSIFORMER_GPU_")}
        environment.update(PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES=str(args.device_bytes),
            PSIFORMER_GPU_EXTERNAL_RESERVE_BYTES=str(256 * 1024**2),
            PSIFORMER_GPU_STATUS_CHECKS="synchronous", PSIFORMER_GPU_METROPOLIS_BYTES=str(16 * 1024**2),
            PSIFORMER_GPU_ADAM_BYTES=str(256 * 1024**2))
        if args.resident_device_authority:
            environment.update(PSIFORMER_GPU_RESIDENT_DEVICE_AUTHORITY="1",
                PSIFORMER_GPU_RESIDENT_ENERGY_ADAM_BYTES=str(256 * 1024**2))
    with (directory / f"{project}.run.log").open("w") as output:
        subprocess.run([str(app), "run.xml"], cwd=directory, env=environment,
                       stdout=output, stderr=subprocess.STDOUT, timeout=remaining(args), check=True)
    rows = history(directory, project)
    if int(rows[-1]["parameter_version"]) != iterations:
        raise AssertionError("Native BFD model version did not advance to the requested update")
    if backend == "cuda":
        log = (directory / f"{project}.run.log").read_text()
        tokens = ("Resident ENERGY accepted-grid ECP appended:", "final_parameter_download_bytes=0") if args.resident_device_authority else (
            "regular-node real GPU accepted-grid ECP replay", "full_parameter_downloads_per_point=0",
            "CPU derivative fallback=no", "replay_only=1 planned_parameter_workspace_bytes=0")
        for token in tokens:
            if token not in log:
                raise AssertionError("Native GPU BFD run did not exercise the new replay route: " + token)
    return directory / f"{project}.s000.last_checkpoint.h5", rows


def suite(app, backend, directory, args):
    continuous = directory / "continuous"
    continuous.mkdir()
    latest, rows = run(app, continuous, "lih_pp_ecp_continuous", backend, 2, args)
    if len(rows) != 2:
        raise AssertionError("Expected two uninterrupted BFD updates")
    if args.profile_only:
        return latest, rows
    split = directory / "split"
    split.mkdir()
    first, _ = run(app, split, "lih_pp_ecp_checkpoint", backend, 1, args)
    resumed, resumed_rows = run(app, split, "lih_pp_ecp_restart", backend, 2, args, first.name)
    # Reuse the established latest-index -> manifest -> ROOT/shard resolution.
    original_files = envelopes(latest, args.h5dump)
    restored_files = envelopes(resumed, args.h5dump)
    if len(original_files) != len(restored_files) or len(original_files) != 2:
        raise AssertionError("Expected matching ROOT and rank-zero sampler envelopes")
    for original, restored in zip(original_files, restored_files):
        subprocess.run([args.h5diff, str(original), str(restored)], check=True, timeout=remaining(args))
    for key in ("iteration", "sample_version", "parameter_version", "samples", "mean_energy",
                "variance", "gradient_norm", "selected_alpha", "committed"):
        if rows[-1][key] != resumed_rows[-1][key]:
            raise AssertionError("Exact BFD restart history mismatch: " + key)
    return latest, rows


def compare_cpu(gpu_result, cpu_result, args):
    import h5py
    import numpy as np
    gpu_latest, gpu_rows = gpu_result
    cpu_latest, cpu_rows = cpu_result
    for gpu, cpu in zip(gpu_rows, cpu_rows):
        for key in ("mean_energy", "variance", "gradient_norm"):
            if not math.isclose(float(gpu[key]), float(cpu[key]), rel_tol=args.rtol, abs_tol=args.atol):
                raise AssertionError("CPU/GPU native BFD history differs beyond tolerance: " + key)
    gpu_root, _ = envelopes(gpu_latest, args.h5dump)
    cpu_root, _ = envelopes(cpu_latest, args.h5dump)
    with h5py.File(gpu_root) as gpu, h5py.File(cpu_root) as cpu:
        values = "wftrain_checkpoint/model/values"
        np.testing.assert_allclose(gpu[values][()], cpu[values][()], rtol=args.rtol, atol=args.atol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-app", type=Path, help="Optional matching CPU build for tolerance-based parity")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    parser.add_argument("--walkers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--blocks", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--device-bytes", type=int, default=3 * 1024**3,
                        help="Aggregate admission cap, not a claim about available device memory")
    parser.add_argument("--h5diff", default=shutil.which("h5diff"))
    parser.add_argument("--h5dump", default=shutil.which("h5dump"))
    parser.add_argument("--profile-only", action="store_true", help="Two uninterrupted updates only, no restart claim")
    parser.add_argument("--prepare-only", action="store_true", help="Generate inputs without executing a binary")
    parser.add_argument("--resident-device-authority", action="store_true",
                        help="Bounded all-active DEVICE ECP Adam and same-CUDA HOST authority oracle")
    parser.add_argument("--rtol", type=float, default=2e-7)
    parser.add_argument("--atol", type=float, default=2e-8)
    args = parser.parse_args()
    args.resident_ecp_profile = args.resident_device_authority
    args.deadline = time.monotonic() + 285 if args.resident_ecp_profile else None
    if args.resident_ecp_profile and (args.backend != "cuda" or args.blocks != 1 or args.walkers != 1 or
            args.cpu_app or args.profile_only or args.timeout > 90 or args.device_bytes != 2 * 1024**3):
        parser.error("resident ECP requires CUDA W1/blocks1, full suite, no CPU repeat, <=90s child and exact2GiB cap")
    args.app = args.app.resolve()
    args.source = args.source.resolve()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.device_bytes <= 0:
        parser.error("timeout and device admission cap must be positive and finite")
    if not all(math.isfinite(value) and value >= 0 for value in (args.rtol, args.atol)):
        parser.error("comparison tolerances must be finite and nonnegative")
    if args.cpu_app and args.backend != "cuda":
        parser.error("--cpu-app is meaningful only with --backend cuda")
    if not args.h5diff or not args.h5dump:
        parser.error("h5diff and h5dump are required for the existing exact-restart semantics")
    if args.output:
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=False)
    else:
        args.output = Path(tempfile.mkdtemp(prefix="psiformer_bfd_replay_"))
    protected = [args.source / "tests/molecules/LiH_pp_psiformer_wftrain/wftrain_ecp_checkpoint.xml"]
    protected += [args.source / f"tests/molecules/LiH_pp/{element}.BFD.xml" for element in ("Li", "H")]
    before = {str(path): digest(path) for path in protected}
    print("Real LiH BFD ordinary-locality replay:", args.output, flush=True)
    if args.prepare_only:
        render(protected[0], args.output / "prepared.xml", "lih_pp_ecp_continuous", args.backend, 2, args)
        return
    result = suite(args.app, args.backend, args.output, args)
    if args.resident_ecp_profile:
        import h5py
        import numpy as np
        oracle_args = copy.copy(args)
        oracle_args.resident_device_authority = False
        oracle_args.profile_only = True
        host = args.output / "host_authority"
        host.mkdir()
        oracle = suite(args.app, "cuda", host, oracle_args)
        compare_cpu(result, oracle, args) # Same CUDA sampler, HOST parameter authority.
        actual_root, _ = envelopes(result[0], args.h5dump)
        expected_root, _ = envelopes(oracle[0], args.h5dump)
        def state(path):
            values = {}
            with h5py.File(path) as archive:
                archive.visititems(lambda key, value: values.update({key: value[()]})
                    if isinstance(value, h5py.Dataset) else None)
            return values
        actual, expected = state(actual_root), state(expected_root)
        for suffix in ("/first_moment", "/second_moment"):
            keys = [key for key in actual if key.endswith(suffix)]
            assert len(keys) == 1
            np.testing.assert_allclose(actual[keys[0]], expected[keys[0]], rtol=args.rtol, atol=args.atol)
        parameters = actual["wftrain_checkpoint/model/values"]
        assert parameters.size == 1593857 and np.isfinite(parameters).all()
        log = (args.output / "continuous" / "lih_pp_ecp_continuous.run.log").read_text()
        windows = list(re.finditer(r"Resident DEVICE authority window:.*uploaded_bytes=(\d+).*downloaded_bytes=(\d+).*device_copied_bytes=(\d+)", log))
        assert len(windows) == 2
        assert all(int(row[2]) < 8 * parameters.size for row in windows) # No P mirror/score download.
        mirrors = list(re.finditer(r"cold HOST mirror refresh: attempted_bytes=(\d+), version=(\d+)", log))
        assert len(mirrors) == 1 and mirrors[0].start() > windows[-1].end()
        assert tuple(map(int, mirrors[0].groups())) == (8 * parameters.size, 2)
        moments = list(re.finditer(r"Resident DEVICE Adam HOST moments before sampling: actual_bytes=(\d+), cold_allowance=(\d+)", log))
        assert len(moments) == 2 and all(tuple(map(int, row.groups())) == (0, 16 * parameters.size) for row in moments)
        appends = list(re.finditer(r"Resident ENERGY accepted-grid ECP appended: points=12, prepared_score_capacity=4, weighted_virtual_tiles=3, weighted_virtual_configurations=12, last_virtual_tile_count=4, final_parameter_download_bytes=0", log))
        assert len(appends) == 2
    if args.cpu_app:
        cpu = args.output / "cpu"
        cpu.mkdir()
        compare_cpu(result, suite(args.cpu_app.resolve(), "cpu", cpu, args), args)
    if before != {str(path): digest(path) for path in protected}:
        raise AssertionError("Native check changed a protected template or BFD asset")
    report = dict(backend=args.backend, app_sha256=digest(args.app), pseudopotential="BFD",
                  walkers=args.walkers, blocks=args.blocks, updates=2,
                  device_admission_cap=args.device_bytes if args.backend == "cuda" else None,
                  exact_restart=not args.profile_only, cpu_parity=bool(args.cpu_app), protected_sha256=before)
    if args.resident_ecp_profile:
        report.update(device_authority=True, all_active_count=1593857, same_cuda_host_authority_oracle=True,
                      warm_transfer_windows=[tuple(map(int, row.groups())) for row in windows],
                      no_warm_host_mirror=True, resident_ecp_q12_cap4_tiles3=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS:", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
