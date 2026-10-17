#!/usr/bin/env python3
"""Finite CPU paired-orbital MSE update/inference/exact-restart smoke test."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "H2_ae_conventional_wftrain"))
from check_sj_training import compare_archives, datasets, text_value


def digest(path):
    """Fingerprint immutable inputs and checkpoint evidence without copying HDF."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qmcpack", required=True, type=Path)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("wftrain_complex_orbital_mse.xml"))
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--energy-reference-profile", action="store_true",
                        help="Run only one bounded two-iteration CPU ENERGY/reference pilot")
    args = parser.parse_args()
    executable, fixture = args.qmcpack.resolve(), args.fixture.resolve()
    root = args.workdir or Path(tempfile.mkdtemp(prefix="complex_orbital_mse_"))
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + (60 if args.energy_reference_profile else 120)
    report = {"workdir": str(root), "executable": str(executable), "app_sha256": digest(executable),
              "fixture_sha256": digest(fixture), "scope": "CPU complex fixed-gauge FreeOrbital orbital-MSE only"}
    if args.energy_reference_profile:
        report["scope"] = "CPU paired ENERGY/reference pilot only; no restart qualification"
    print(f"Evidence: {root}", flush=True)
    active = None

    def interrupted(signum, frame):
        """Do not leave an owned native child behind when the checker is stopped."""
        if active is not None and active.poll() is None:
            try:
                os.killpg(active.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            active.wait()
        raise SystemExit(f"checker interrupted by signal {signum}")

    previous = {number: signal.signal(number, interrupted) for number in (signal.SIGTERM, signal.SIGINT)}

    def run(label, iterations=2, restart=None, mutate=None, rejection=None):
        """Run once inside the aggregate deadline and retain the first failure."""
        nonlocal active
        directory = root / label
        directory.mkdir()
        tree = ET.parse(fixture)
        training = tree.find("qmc/training")
        if args.energy_reference_profile:
            training.set("objective", "energy")
            for name in ("teacher", "target_provenance", "orbital_target"):
                training.remove(training.find(f"parameter[@name='{name}']"))
            training.find("parameter[@name='line_search']").text = "reference"
            for name, value in (("line_search_minimum_ess_fraction", ".5"),
                                ("line_search_maximum_weight", ".5"), ("line_search_confidence", "2")):
                ET.SubElement(training, "parameter", name=name).text = value
            tree.find("qmc/parameter[@name='walkers_per_rank']").text = "2"
            tree.find("qmc/parameter[@name='blocks']").text = "2"
        training.find("parameter[@name='iterations']").text = str(iterations)
        if restart is not None:
            ET.SubElement(training, "parameter", name="restart_path").text = str(restart.resolve())
        if mutate:
            mutate(tree)
        tree.write(directory / "run.xml")
        if args.energy_reference_profile:
            report["derived_xml_sha256"] = digest(directory / "run.xml")
        remaining = min(60 if args.energy_reference_profile else 45, deadline - time.monotonic())
        assert remaining > 0, "aggregate 120-second deadline exhausted"
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (directory / "console.log").open("w") as output:
            command = [str(executable), "run.xml"]
            if args.energy_reference_profile:
                command = ["/usr/bin/time", "-v", "-o", "resource_time.txt", *command]
            process = subprocess.Popen(command, cwd=directory, env=env,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            active = process
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                (directory / "run.exit").write_text(str(process.returncode) + "\n")
                raise AssertionError(f"{label}: bounded process timeout; see {directory}")
        (directory / "run.exit").write_text(str(code) + "\n")
        console = (directory / "console.log").read_text()
        if rejection:
            assert code != 0 and rejection in console, (label, code, console[-6000:])
            assert not list(directory.rglob("*.parameters.h5")), directory
            return directory
        assert code == 0, (label, code, console[-6000:])
        assert ("WFTrain paired CPU prospective named numeric bytes plus allowance=" if args.energy_reference_profile
                else "paired CPU orbital-MSE score owner=") in console, directory
        with (directory / "complex_orbital_mse.s000.training_info.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == (iterations if restart is None else 1), (label, rows)
        if args.energy_reference_profile:
            receipts = [tuple(map(float, values)) for values in re.findall(
                r"WFTrain candidate alpha (\S+) loss (\S+) loss_difference (\S+) "
                r"difference_error (\S+) ess (\S+) maximum_weight (\S+) valid ([01])", console)]
            assert len(receipts) == 6 * len(rows), receipts
            previous_values, previous_state = None, None
            models = sorted(directory.rglob("*.parameters.h5"))
            assert len(models) == len(rows), models
            for ordinal, row in enumerate(rows):
                candidates = receipts[6*ordinal:6*(ordinal+1)]
                assert {value[0] for value in candidates} == {0, .125, .25, .5, .75, 1}, candidates
                assert all(math.isfinite(value) for candidate in candidates for value in candidate), candidates
                assert all(0 < value[4] <= 4 and 0 < value[5] <= 1 for value in candidates), candidates
                alpha = float(row["selected_alpha"])
                committed = int(row["committed"])
                assert alpha in {value[0] for value in candidates} and committed == int(alpha > 0), row
                assert alpha == 0 or any(value[0] == alpha and value[6] == 1 and
                    value[2] + 2*value[3] < 0 for value in candidates), row
                assert row["objective"] == "energy" and int(row["samples"]) == 4, row
                assert int(row["parameter_version"]) == int(row["sample_version"]) + committed, row
                assert all(math.isfinite(float(row[key])) for key in ("mean_energy", "variance", "gradient_norm")), row
                values = datasets(models[ordinal])["values"]
                bundle = models[ordinal].parent / (models[ordinal].name.removesuffix(".parameters.h5") + ".checkpoint")
                state = datasets(next(bundle.glob("*.state.h5")))
                keys = ["wftrain_checkpoint/optimizer/" + key for key in
                        ("first_moment", "second_moment", "accepted_update_count")]
                assert all(key in state for key in keys) and np.isfinite(values).all()
                if alpha == 0:
                    if ordinal == 0:
                        assert int(row["parameter_version"]) == 0, row
                    if previous_values is not None:
                        np.testing.assert_array_equal(values, previous_values)
                    for key in keys:
                        np.testing.assert_array_equal(state[key], previous_state[key] if previous_state else np.zeros_like(state[key]))
                previous_values, previous_state = values, state
        for row in rows:
            if args.energy_reference_profile:
                continue
            assert row["objective"] == "orbital_mse" and row["committed"] == "1", row
            assert int(row["samples"]) == 1, row
            assert int(row["parameter_version"]) == int(row["sample_version"]) + 1, row
            assert all(math.isfinite(float(row[name])) for name in ("orbital_loss", "gradient_norm")), row
            assert float(row["orbital_loss"]) >= 0 and float(row["gradient_norm"]) > 0, row
        scalars = sorted((directory / "complex_orbital_mse.s000.inference").glob("*.scalar.dat"))
        assert len(scalars) == len(rows), (label, scalars)
        for scalar in scalars:
            text = scalar.read_text()
            assert "complete yes" in text, scalar
            header = next(line.split()[1:] for line in text.splitlines()
                          if line.startswith("#") and "BlockWeight" in line)
            data = [line.split() for line in text.splitlines() if line.strip() and not line.startswith("#")]
            assert len(data) == 1 and all(math.isfinite(float(value)) for value in data[0]), scalar
            assert float(data[0][header.index("BlockWeight")]) == 1, scalar
        assert (directory / "complex_orbital_mse.s000.last_checkpoint.h5").is_file()
        return directory, rows

    try:
        continuous, rows = run("continuous")
        if args.energy_reference_profile:
            assert digest(fixture) == report["fixture_sha256"], "source fixture changed"
            report["maximum_rss_kib"] = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)",
                (continuous / "resource_time.txt").read_text()).group(1))
            report.update(status="PASS", iterations=2, samples_per_iteration=4,
                          selected_alphas=[float(row["selected_alpha"]) for row in rows],
                          first_rejection_initial_parameter_equality="not independently observable",
                          exact_restart_qualified=False)
            return
        split, _ = run("split", iterations=1)
        checkpoint = split / "complex_orbital_mse.s000.last_checkpoint.h5"
        protected = {str(path): digest(path) for path in split.rglob("*.h5")}
        resumed, restored_rows = run("resumed", restart=checkpoint)
        ignored = {"sampling_seconds", "total_seconds", "resident_bytes"}
        assert all(rows[-1][key] == restored_rows[0][key] for key in rows[-1] if key not in ignored)
        models = sorted(continuous.rglob("*.parameters.h5"))
        restored = sorted(resumed.rglob("*.parameters.h5"))
        assert len(models) == 2 and len(restored) == 1
        lhs, rhs = datasets(models[-1]), datasets(restored[-1])
        assert lhs.keys() == rhs.keys()
        for key in lhs:
            np.testing.assert_array_equal(lhs[key], rhs[key], err_msg=f"exact restart model: {key}")
        first = datasets(models[0])
        assert np.isfinite(lhs["values"]).all() and np.any(lhs["values"] != first["values"])
        names = [text_value(value) for value in np.asarray(lhs["layout/names"]).reshape(-1)]
        offsets = np.asarray(lhs["layout/offsets"]).reshape(-1)
        imaginary = [slice(int(offsets[i]), int(offsets[i + 1])) for i, name in enumerate(names) if name == "w_imag"]
        assert imaginary and any(np.any(lhs["values"][part] != first["values"][part]) for part in imaginary)
        for pattern in ("*.state.h5", "*.rank_*.h5"):
            left = models[-1].parent / (models[-1].name.removesuffix(".parameters.h5") + ".checkpoint")
            right = restored[-1].parent / (restored[-1].name.removesuffix(".parameters.h5") + ".checkpoint")
            compare_archives(left, right, pattern)
        final_scalar = sorted((continuous / "complex_orbital_mse.s000.inference").glob("*.scalar.dat"))[-1]
        restored_scalar = sorted((resumed / "complex_orbital_mse.s000.inference").glob("*.scalar.dat"))[-1]
        def physical_row(path):
            text = path.read_text().splitlines()
            header = next(line.split()[1:] for line in text if line.startswith("#") and "BlockWeight" in line)
            values = next(line.split() for line in text if line.strip() and not line.startswith("#"))
            assert len(header) == len(values)
            # Wall-clock CPU time is not a physical inference result or restart state.
            return header, {name: value for name, value in zip(header, values) if name != "BlockCPU"}

        assert physical_row(final_scalar) == physical_row(restored_scalar), "exact restart inference"

        def teacher_twist(tree):
            tree.find(".//wavefunction[@name='teacher']//sposet").set("twist", "0.18 -0.11 0.07")

        run("reject_teacher_twist", restart=checkpoint, mutate=teacher_twist,
            rejection="Complex orbital target twist mismatch")

        def changed_cell(tree):
            tree.find(".//simulationcell/parameter[@name='lattice']").text = "8.1 0 0 0.8 8 0 0.4 0.6 8"

        run("reject_restart_cell", restart=checkpoint, mutate=changed_cell,
            rejection="Training checkpoint stage identity or configuration mismatch")
        assert protected == {str(path): digest(path) for path in split.rglob("*.h5")}
        report.update(status="PASS", committed_updates=2, samples_per_update=1,
                      exact_model_optimizer_sampler_restart=True, imaginary_head_changed=True)
    except BaseException as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
