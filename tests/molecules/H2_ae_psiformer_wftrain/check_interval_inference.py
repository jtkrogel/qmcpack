#!/usr/bin/env python3
"""Tiny native H2 regression for inference cadence and exact training isolation."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def datasets(path):
    result = {}
    with h5py.File(path, "r") as handle:
        handle.visititems(lambda name, obj: result.update({name: obj[()]})
                          if isinstance(obj, h5py.Dataset) else None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--ranks", type=int, default=1)
    args = parser.parse_args()
    executable = args.executable.resolve()
    fixture = args.fixture.resolve()
    work = Path(tempfile.mkdtemp(prefix="wftrain_inference_verify_"))
    print(work, flush=True)

    def run(label, inference=None, iterations=3, restart=None, root="case", rejected=False, collision=False,
            max_seconds=None, trigger_stop=False):
        directory = work / label
        directory.mkdir()
        tree = ET.parse(fixture)
        tree.find("project").set("id", root)
        if max_seconds is not None:
            ET.SubElement(tree.find("project"), "parameter", name="max_seconds").text = str(max_seconds)
        tree.find("wavefunction").set("print_parameters", "no")
        qmc = tree.find("qmc")
        training = qmc.find("training")
        training.find("parameter[@name='iterations']").text = str(iterations)
        if rejected:
            training.find("parameter[@name='learning_rate']").text = "1e-4"
            for name, value in {"line_search": "reference", "line_search_minimum_ess_fraction": "1",
                                "line_search_maximum_weight": "1", "line_search_confidence": "0"}.items():
                ET.SubElement(training, "parameter", name=name).text = value
        if restart:
            ET.SubElement(training, "parameter", name="restart_path").text = str(restart)
        if inference:
            interval, last = inference
            node = ET.SubElement(qmc, "inference", interval=str(interval), last=last, move="pbyp")
            for name, value in {"walkers_per_rank": 2, "crowds": 1, "warmupsteps": 1,
                                "blocks": 1000 if trigger_stop else 2, "steps": 1, "substeps": 2,
                                "timestep": "1e-4", "usedrift": "yes"}.items():
                ET.SubElement(node, "parameter", name=name).text = str(value)
            ET.SubElement(ET.SubElement(node, "estimators"), "unknown")
        tree.write(directory / "run.xml")
        if collision:
            family = directory / f"{root}.s000.inference"
            family.mkdir()
            future = family / f"{root}.s000.i0000002.scalar.dat"
            future.write_text("preserve existing inference data\n")
            checksum = hashlib.sha256(future.read_bytes()).digest()
        command = [str(executable), "run.xml"]
        if args.ranks > 1:
            command = ["mpirun", "-np", str(args.ranks)] + command
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (directory / "run.log").open("w") as log:
            if trigger_stop:
                result = subprocess.Popen(command, cwd=directory, env=environment, stdout=log,
                                          stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 60
                while not list(directory.glob(f"{root}.s000.inference/*.scalar.dat")):
                    assert result.poll() is None and time.monotonic() < deadline, directory
                    time.sleep(0.01)
                (directory / f"{root}.STOP").write_text("stop during inference\n")
                result.wait(timeout=60)
            else:
                result = subprocess.run(command, cwd=directory, env=environment, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=180)
        if collision:
            assert result.returncode != 0, directory
            assert checksum == hashlib.sha256(future.read_bytes()).digest()
        else:
            assert result.returncode == 0, (directory, (directory / "run.log").read_text()[-4000:])
        return directory

    off = run("off")
    on = run("on", (2, "no"))
    last = run("last", (2, "yes"))
    due_final = run("due_final", (2, "yes"), iterations=2)
    rejected = run("rejected", (2, "no"), rejected=True)
    assert not (off / "case.s000.inference").exists()
    for directory, indices in ((on, [1]), (last, [1, 2]), (due_final, [1]), (rejected, [1])):
        family = directory / "case.s000.inference"
        assert sorted(p.name for p in family.iterdir()) == [f"case.s000.i{i:07d}.scalar.dat" for i in indices]
        assert not list(directory.rglob("*.stat.h5"))
        assert not (directory / "case.s000.scalar.dat").exists()
        for index in indices:
            text = (family / f"case.s000.i{index:07d}.scalar.dat").read_text()
            rows = [line.split() for line in text.splitlines() if line.strip() and not line.startswith("#")]
            assert len(rows) == 2, text
            header = text.splitlines()[0].split()[1:]
            assert all(float(row[header.index("BlockWeight")]) == 2 * args.ranks for row in rows)
            assert f"iteration {index} parameter_version " in text and "complete yes" in text
        assert "ignores supplemental estimators" in (directory / "run.log").read_text()
    for pattern in ("*.parameters.h5", "*.state.h5", "*.rank_*.h5"):
        first = sorted((off / "case.s000.training").rglob(pattern))
        second = sorted((on / "case.s000.training").rglob(pattern))
        assert first and len(first) == len(second), pattern
        for a, b in zip(first, second):
            left, right = datasets(a), datasets(b)
            assert left.keys() == right.keys()
            for name in left:
                np.testing.assert_array_equal(left[name], right[name], err_msg=f"{pattern}: {name}")
    def history(directory):
        with (directory / "case.s000.training_info.csv").open() as handle:
            return list(csv.DictReader(handle))
    for left, right in zip(history(off), history(on)):
        for name in ("iteration", "samples", "mean_energy", "variance", "gradient_norm", "selected_alpha", "committed"):
            assert left[name] == right[name], (name, left[name], right[name])
    assert len(history(on)) == 3
    assert any(row["committed"] == "0" for row in history(rejected))
    split = run("split", (2, "no"), iterations=2)
    restart = split / "case.s000.last_checkpoint.h5"
    branch = run("branch", (2, "yes"), iterations=3, restart=restart, root="branch")
    assert sorted(p.name for p in (branch / "branch.s000.inference").iterdir()) == ["branch.s000.i0000002.scalar.dat"]
    run("collision", (2, "yes"), restart=restart, collision=True)
    budget = run("budget", (1, "yes"), max_seconds=1)
    assert not (budget / "case.s000.inference").exists()
    assert "complete no skipped runtime_stop" in (budget / "run.log").read_text()
    assert len(history(budget)) == 1 and (budget / "case.s000.last_checkpoint.h5").exists()
    partial = run("partial", (1, "yes"), trigger_stop=True)
    text = (partial / "case.s000.inference/case.s000.i0000000.scalar.dat").read_text()
    assert "requested_blocks 1000 complete no" in text
    assert len(history(partial)) == 1 and (partial / "case.s000.last_checkpoint.h5").exists()
    print("PASS: cadence, last/no-last, accepted/rejected labels, scalar-only output, exact parameter/optimizer/walker/RNG isolation and cumulative restart")


if __name__ == "__main__":
    main()
