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
    args = parser.parse_args()
    executable = args.executable.resolve()
    fixture = args.fixture.resolve()
    work = Path(tempfile.mkdtemp(prefix="conventional_sj_verify_"))
    print(work, flush=True)

    def run(label, *, optimizer="adam", inference=True, iterations=2, restart=None, mutate=None, reject=False):
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
        with (directory / "run.log").open("w") as log:
            result = subprocess.run([str(executable), "run.xml"], cwd=directory, env=environment,
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
            assert row["committed"] == "1", (directory, row)
            assert int(row["samples"]) == 4, (directory, row)
            assert all(math.isfinite(float(row[name])) for name in ("mean_energy", "variance", "gradient_norm")), row
            assert float(row["gradient_norm"]) > 0, row
            assert int(row["parameter_version"]) == int(row["sample_version"]) + 1, row
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
            assert float(data[0][header.index("BlockWeight")]) == 2, path
        models = sorted((directory / "sj.s000.training").rglob("*.parameters.h5"))
        assert len(models) == expected_rows, (directory, models)
        initial = [float(value) for node in tree.findall("wavefunction/jastrow/correlation/coefficients")
                   for value in node.text.split()]
        previous = np.asarray(initial)
        fixed = tree.find(".//sposet/coefficient")
        for model in models:
            with h5py.File(model, "r") as handle:
                group = handle["conventional_wavefunction"]
                parameters = np.asarray(group["parameters"][()]).reshape(-1)
                assert parameters.size == 8 and np.isfinite(parameters).all(), model
                assert not np.array_equal(parameters, previous), model
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

    adam = run("adam")
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
        run("reject_kfac", optimizer="kfac", reject="requires all-electron ENERGY Adam/plain SGD")
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
