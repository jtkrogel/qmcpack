#!/usr/bin/env python3
"""Bounded native SJ update/inference regression; retain all evidence in a temp dir."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def datasets(path):
    result = {}
    with h5py.File(path, "r") as handle:
        handle.visititems(lambda name, obj: result.update({name: obj[()]})
                          if isinstance(obj, h5py.Dataset) else None)
    return result


def compare_archives(left, right, pattern):
    first = sorted(left.rglob(pattern))
    second = sorted(right.rglob(pattern))
    assert first and len(first) == len(second), (pattern, first, second)
    for a, b in zip(first, second):
        lhs, rhs = datasets(a), datasets(b)
        assert lhs.keys() == rhs.keys(), (a, b)
        for name in lhs:
            np.testing.assert_array_equal(lhs[name], rhs[name], err_msg=f"{pattern}: {name}")


def history(directory):
    with (directory / "sj.s000.training_info.csv").open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("fixture", type=Path, nargs="?", default=Path(__file__).with_name("wftrain_sj.xml"))
    parser.add_argument("--restart", action="store_true", help="also require split-run exact continuation")
    parser.add_argument("--preflight", action="store_true", help="also check unsupported model/optimizer rejection")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--profile-only", action="store_true", help="one bounded Adam/inference case for MPI/OpenMP")
    parser.add_argument("--optimizer-screen", action="store_true", help="six bounded nonlinear SJ optimizer cases")
    parser.add_argument("--family-screen", action="store_true", help="bounded LM/ALM/PII/minPII cases; report rejection honestly")
    args = parser.parse_args()
    if args.threads <= 0 or args.ranks <= 0:
        parser.error("--threads and --ranks must be positive")
    if (args.profile_only or args.optimizer_screen or args.family_screen) and (args.restart or args.preflight):
        parser.error("profile/screen options select their cases; omit --restart/--preflight")
    if sum((args.profile_only, args.optimizer_screen, args.family_screen)) > 1:
        parser.error("select one of --profile-only, --optimizer-screen, --family-screen")
    executable = args.executable.resolve()
    fixture = args.fixture.resolve()
    work = Path(tempfile.mkdtemp(prefix="conventional_sj_verify_"))
    print(work, flush=True)

    def run(label, *, optimizer="adam", inference=True, iterations=2, restart=None, mutate=None, reject=False,
            samples_per_rank=4, require_committed=True):
        directory = work / label
        directory.mkdir()
        tree = ET.parse(fixture)
        tree.find("project").set("id", "sj")
        qmc = tree.find("qmc")
        training = qmc.find("training")
        training.set("optimizer", optimizer)
        training.find("parameter[@name='iterations']").text = str(iterations)
        if not inference:
            qmc.remove(qmc.find("inference"))
        if restart is not None:
            ET.SubElement(training, "parameter", name="restart_path").text = str(restart)
        if mutate:
            mutate(tree)
        # No generated recipe is allowed to unfreeze the Slater implicitly.
        if not reject:
            assert all(node.get("optimize") == "no" for node in tree.findall(".//sposet"))
        tree.write(directory / "run.xml")
        environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1")
        command = [str(executable), "run.xml"]
        if args.ranks > 1:
            command = [args.mpiexec, "-np", str(args.ranks)] + command
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=directory, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180)
        text = (directory / "run.log").read_text()
        if reject:
            assert result.returncode != 0, (directory, text[-4000:])
            assert reject in text, (directory, reject, text[-4000:])
            assert not list(directory.glob("*.training/*/*.state.h5")), directory
            return directory
        assert result.returncode == 0, (directory, text[-6000:])
        rows = history(directory)
        expected_rows = iterations if restart is None else 1
        assert len(rows) == expected_rows, (directory, rows)
        for row in rows:
            if require_committed:
                assert row["committed"] == "1", (directory, row)
            assert row["committed"] in ("0", "1"), (directory, row)
            assert int(row["samples"]) == samples_per_rank * args.ranks, (directory, row)
            assert all(math.isfinite(float(row[name])) for name in ("mean_energy", "variance", "gradient_norm")), row
            if require_committed:
                assert float(row["gradient_norm"]) > 0, row
            assert int(row["parameter_version"]) == int(row["sample_version"]) + int(row["committed"]), row
        assert (directory / "sj.s000.last_checkpoint.h5").is_file(), directory
        family = directory / "sj.s000.inference"
        files = sorted(family.glob("*.scalar.dat"))
        assert len(files) == (expected_rows if inference else 0), (directory, files)
        for path, row in zip(files, rows):
            scalar = path.read_text()
            iteration = int(row["iteration"])
            assert f"iteration {iteration} parameter_version {row['parameter_version']} " in scalar, path
            assert "complete yes" in scalar, path
            data = [line.split() for line in scalar.splitlines() if line.strip() and not line.startswith("#")]
            assert len(data) == 1, path
            header = next(line.split()[1:] for line in scalar.splitlines() if line.startswith("#") and "BlockWeight" in line)
            assert float(data[0][header.index("BlockWeight")]) == 2 * args.ranks, path
        models = sorted((directory / "sj.s000.training").rglob("*.parameters.h5"))
        assert len(models) == expected_rows, (directory, models)
        initial = [float(value) for node in tree.findall("wavefunction/jastrow/correlation/coefficients")
                   for value in node.text.split()]
        previous = np.asarray(initial)
        fixed = tree.find(".//sposet/coefficient")
        for model, row in zip(models, rows):
            with h5py.File(model, "r") as handle:
                group = handle["conventional_wavefunction"]
                parameters = np.asarray(group["parameters"][()]).reshape(-1)
                assert parameters.size == 8 and np.isfinite(parameters).all(), model
                if row["committed"] == "1":
                    assert not np.array_equal(parameters, previous), model
                elif restart is None:
                    np.testing.assert_array_equal(parameters, previous, err_msg=f"rejected model: {model}")
                previous = parameters
                serialized = group["wavefunction_xml"][()]
                if isinstance(serialized, np.ndarray):
                    serialized = serialized.reshape(-1)[0]
                if isinstance(serialized, bytes):
                    serialized = serialized.decode()
                exported = ET.fromstring(serialized)
                coefficients = exported.find(".//sposet/coefficient")
                assert coefficients.attrib == fixed.attrib, model
                assert coefficients.text.split() == fixed.text.split(), model
                assert all(node.get("optimize") == "no" for node in exported.findall(".//sposet")), model
        return directory

    if args.family_screen:
        reports = []
        for optimizer in ("lm", "alm", "pii", "minpii"):
            def controls(tree):
                qmc = tree.find("qmc")
                qmc.find("parameter[@name='walkers_per_rank']").text = "4"
                qmc.find("parameter[@name='blocks']").text = "8"
                training = qmc.find("training")
                limits = {"damping": "0.01", "krylov_maximum_iterations": "64",
                          "krylov_maximum_operator_applications": "256", "krylov_maximum_seconds": "15",
                          "krylov_relative_tolerance": "1e-6"}
                if optimizer in ("lm", "alm"):
                    limits.update(lm_maximum_subspace_dimension="9", lm_inner_iterations="32",
                                  lm_inner_restart_length="9")
                else:
                    limits.update(gmres_restart_length="16", maximum_damping_retries="0",
                                  pii_energy_undershoot="1.0")
                for name, value in limits.items():
                    ET.SubElement(training, "parameter", name=name).text = value
            try:
                directory = run(optimizer, optimizer=optimizer, mutate=controls,
                                samples_per_rank=32, require_committed=False)
                rows = history(directory)
                decisions = ",".join(row["committed"] for row in rows)
                reports.append((optimizer, "complete", decisions))
                print(f"{optimizer}: complete; committed={decisions}", flush=True)
            except (AssertionError, subprocess.TimeoutExpired) as error:
                reports.append((optimizer, "failed", ""))
                print(f"{optimizer}: FAILED; {error}", flush=True)
        with (work / "family_results.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("optimizer", "status", "committed"))
            writer.writerows(reports)
        assert all(status == "complete" for _, status, _ in reports), work
        print("PASS: bounded LM/ALM/PII/minPII physical-state consistency; decisions recorded without forcing acceptance")
        return

    if args.optimizer_screen:
        for optimizer in ("momentum_sgd", "rmsprop", "amsgrad", "sr", "minsr", "spring"):
            def controls(tree):
                if optimizer in ("sr", "minsr", "spring"):
                    training = tree.find("qmc/training")
                    limits = {"damping": "0.01", "krylov_maximum_iterations": "32",
                              "krylov_maximum_operator_applications": "128", "krylov_maximum_seconds": "10",
                              "krylov_relative_tolerance": "1e-8"}
                    if optimizer == "sr":
                        limits["maximum_damping_attempts"] = "1"
                    for name, value in limits.items():
                        ET.SubElement(training, "parameter", name=name).text = value
            directory = run(optimizer, optimizer=optimizer, mutate=controls)
            for row in history(directory):
                for name, maximum in (("krylov_iterations", 32), ("krylov_operator_applications", 128)):
                    if row.get(name):
                        assert int(row[name]) <= maximum, (optimizer, row)
        print("PASS: bounded momentum_sgd/rmsprop/amsgrad/sr/minsr/spring SJ updates and inference")
        return

    adam = run("adam")
    if args.profile_only:
        print(f"PASS: bounded SJ Adam/inference {args.ranks} MPI x {args.threads} OpenMP; accepted model export")
        return
    off = run("adam_no_inference", inference=False)
    sgd = run("sgd", optimizer="sgd")
    for lhs, rhs in zip(history(adam), history(off)):
        for name in ("iteration", "samples", "mean_energy", "variance", "gradient_norm", "selected_alpha", "committed"):
            assert lhs[name] == rhs[name], (name, lhs[name], rhs[name])
    for pattern in ("*.parameters.h5", "*.state.h5", "*.rank_*.h5"):
        compare_archives(adam / "sj.s000.training", off / "sj.s000.training", pattern)
    if args.restart:
        split = run("split", iterations=1)
        resumed = run("resumed", restart=split / "sj.s000.last_checkpoint.h5")
        for name in ("samples", "mean_energy", "variance", "gradient_norm", "selected_alpha", "committed"):
            assert history(adam)[-1][name] == history(resumed)[0][name], name
        # Compare the final model rather than wall-time/counter metadata.
        final = sorted((adam / "sj.s000.training").rglob("*.parameters.h5"))[-1]
        restored = sorted((resumed / "sj.s000.training").rglob("*.parameters.h5"))[-1]
        lhs, rhs = datasets(final), datasets(restored)
        assert lhs.keys() == rhs.keys()
        for name in lhs:
            np.testing.assert_array_equal(lhs[name], rhs[name], err_msg=f"restart model: {name}")
    if args.preflight:
        run("reject_kfac", optimizer="kfac", reject="no affine observations required by KFAC")
        run("reject_zero_cap", reject="legacy_maximum_parameters must be positive", mutate=lambda tree: setattr(
            tree.find("qmc/training/parameter[@name='legacy_maximum_parameters']"), "text", "0"))

        def enable_orbitals(tree):
            spo = tree.find("wavefunction/sposet_collection/sposet")
            spo.set("optimize", "yes")
            spo.set("size", "2")
            spo.find("coefficient").set("size", "2")
            spo.find("coefficient").text = "0.55 0.55  0.55 -0.55"
        run("reject_active_spo", reject="only fixed-Slater J1/J2 parameters", mutate=enable_orbitals)
    completed = "Adam/SGD SJ updates, interval inference, frozen Slater, exact inference isolation"
    if args.restart:
        completed += ", exact split restart"
    if args.preflight:
        completed += ", expected preflight diagnostics"
    print("PASS: " + completed)


if __name__ == "__main__":
    main()
