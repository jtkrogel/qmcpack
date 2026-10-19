#!/usr/bin/env python3
"""Bounded CPU complex KFAC XML updates and exact model/optimizer/sampler restart."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "H2_ae_conventional_wftrain"))
from check_sj_training import compare_archives, datasets, text_value


def digest(path):
    """Identify the exact application and generated XML used by this test."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter(parent, name, value):
    """Set an existing XML parameter or add one explicit control."""
    node = parent.find(f"parameter[@name='{name}']")
    if node is None:
        node = ET.SubElement(parent, "parameter", name=name)
    node.text = str(value)


def main():
    """Run finite Gamma/twist cases and retain exact full/split restart evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qmcpack", type=Path, required=True)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--ranks", type=int, default=1, choices=(1, 2))
    parser.add_argument("--twist-only", action="store_true")
    parser.add_argument("--real-limit", action="store_true")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--clipped", action="store_true")
    policy.add_argument("--defaults", choices=("deepqmc", "deepsolid", "jaqmc"))
    args = parser.parse_args()
    if args.real_limit and not args.defaults:
        parser.error("--real-limit requires an explicit defaults policy")
    application = args.qmcpack.resolve()
    root = args.workdir or Path(tempfile.mkdtemp(prefix="complex_kfac_"))
    root.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).with_name("wftrain_complex_orbital_mse.xml")
    preset_objectives = dict(deepqmc="clipped_covariance", deepsolid="componentwise_mean_residual",
                             jaqmc="componentwise_mad_covariance")
    report = dict(application=str(application), application_sha256=digest(application),
                  fixture_sha256=digest(fixture), ranks=args.ranks, cases=[],
                  objective=preset_objectives[args.defaults] if args.defaults else
                      "componentwise_mad_covariance" if args.clipped else "raw_covariance",
                  defaults=args.defaults)
    print(f"Evidence: {root}", flush=True)

    def run(label, twist, iterations, restart=None, clip_width=5, rejection=None, defaults=None, rate_override=None):
        """Run one tiny native energy workflow with actual complex local energies."""
        directory = root / label
        directory.mkdir()
        tree = ET.parse(fixture)
        system = tree.find("qmcsystem")
        system.remove(system.find("wavefunction[@name='teacher']"))
        student = system.find("wavefunction/psiformer")
        student.set("twist", twist)
        student.set("optimize", "yes")
        student.set("optimize_scope", "all")
        if args.real_limit:
            real_fixture = ET.parse(Path(__file__).with_name("wftrain.xml"))
            student.set("initialization", real_fixture.find(".//psiformer").get("initialization"))
            student.attrib.pop("max_clone_numeric_bytes", None)
        qmc = tree.find("qmc")
        inference = qmc.find("inference")
        if inference is not None:
            qmc.remove(inference)
        training = qmc.find("training")
        training.set("objective", "energy")
        training.set("optimizer", "kfac")
        for name in ("teacher", "target_provenance", "orbital_target"):
            training.remove(training.find(f"parameter[@name='{name}']"))
        controls = dict(iterations=iterations, learning_rate="1e-7", damping="0.1",
                        maximum_retained_bytes=2147483648, kfac_persist_state="yes",
                        kfac_ema_decay="0.5", kfac_inverse_period=2,
                        kfac_curvature="registered_gaussian", kfac_objective="raw_covariance",
                        kfac_damping_policy="trace_balanced")
        if args.clipped:
            controls.update(kfac_objective="componentwise_mad_covariance", kfac_complex_clip_width=clip_width)
        if args.defaults:
            for child in list(training):
                if child.get("name") not in ("checkpoint_interval", "parameter_chunk_size", "line_search"):
                    training.remove(child)
            controls = dict(defaults=defaults or args.defaults, iterations=iterations, maximum_retained_bytes=2147483648)
            if clip_width != 5:
                controls["kfac_clip_width" if args.defaults == "deepqmc" else "kfac_complex_clip_width"] = clip_width
        if rate_override is not None:
            controls["learning_rate"] = rate_override
        for name, value in controls.items():
            parameter(training, name, value)
        if restart is not None:
            parameter(training, "restart_path", restart.resolve())
        parameter(qmc, "walkers_per_rank", 2)
        parameter(qmc, "timestep", "0.01")
        tree.write(directory / "run.xml", encoding="utf-8", xml_declaration=True)
        command = [str(application), "run.xml"]
        if args.ranks > 1:
            command = ["mpiexec", "--bind-to", "none", "-n", str(args.ranks), *command]
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (directory / "console.log").open("w") as output:
            result = subprocess.run(command, cwd=directory, env=environment, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=90)
        console = (directory / "console.log").read_text()
        if rejection is not None:
            assert result.returncode != 0 and rejection in console, (label, result.returncode, console[-6000:])
            assert not list(directory.rglob("*.parameters.h5")), directory
            return directory, []
        assert result.returncode == 0, (label, result.returncode, console[-6000:])
        assert "parameter_response_mask=0" in console
        if args.clipped or args.defaults:
            assert "WFTrain score weighting:" in console
        if args.defaults:
            convention = dict(deepqmc="half_covariance", deepsolid="real_centered_residual", jaqmc="physical_covariance")
            assert f"defaults={args.defaults} policy_version=1 gradient_convention={convention[args.defaults]}" in console
        if rate_override is not None:
            resolved = re.search(r"\blearning_rate=([^\s]+)", console)
            assert resolved and float(resolved.group(1)) == float(rate_override), console[-6000:]
        with (directory / "complex_orbital_mse.s000.training_info.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == (iterations if restart is None else 1), rows
        for row in rows:
            assert row["objective"] == "energy" and row["optimizer"] == "kfac" and row["committed"] == "1", row
            assert int(row["samples"]) == 2 * args.ranks, row
            assert int(row["parameter_version"]) == int(row["sample_version"]) + 1, row
            assert all(math.isfinite(float(row[name])) for name in ("mean_energy", "variance", "gradient_norm")), row
            assert float(row["gradient_norm"]) > 0, row
        return directory, rows

    try:
        twists = [("twist", "0.17 -0.11 0.07")]
        if args.real_limit:
            twists = [("real_gamma", "0 0 0")]
        elif not args.twist_only:
            twists.insert(0, ("gamma", "0 0 0"))
        for label, twist in twists:
            continuous, rows = run(label + "_continuous", twist, 2)
            split, _ = run(label + "_split", twist, 1)
            checkpoint = split / "complex_orbital_mse.s000.last_checkpoint.h5"
            protected = {str(path): digest(path) for path in split.rglob("*.h5")}
            resumed, resumed_rows = run(label + "_resumed", twist, 2, checkpoint)
            ignored = {"sampling_seconds", "total_seconds", "resident_bytes"}
            assert all(rows[-1][key] == resumed_rows[0][key] for key in rows[-1] if key not in ignored)
            full_models = sorted(continuous.rglob("*.parameters.h5"))
            resumed_models = sorted(resumed.rglob("*.parameters.h5"))
            assert len(full_models) == 2 and len(resumed_models) == 1
            left, right = datasets(full_models[-1]), datasets(resumed_models[-1])
            assert left.keys() == right.keys()
            for name in left:
                np.testing.assert_array_equal(left[name], right[name], err_msg=f"exact model restart: {name}")
            assert np.isfinite(left["values"]).all()
            names = [text_value(value) for value in np.asarray(left["layout/names"]).reshape(-1)]
            offsets = np.asarray(left["layout/offsets"]).reshape(-1)
            initial = datasets(full_models[0])["values"]
            imaginary = [slice(int(offsets[i]), int(offsets[i + 1])) for i, name in enumerate(names) if name == "w_imag"]
            if args.real_limit:
                assert not imaginary and np.any(left["values"] != initial)
            else:
                assert imaginary and any(np.any(left["values"][part] != initial[part]) for part in imaginary)
            bundle = lambda model: model.parent / (model.name.removesuffix(".parameters.h5") + ".checkpoint")
            for pattern in ("*.state.h5", "*.rank_*.h5"):
                compare_archives(bundle(full_models[-1]), bundle(resumed_models[-1]), pattern)
            assert len(list(bundle(resumed_models[-1]).glob("*.rank_*.h5"))) == args.ranks
            assert protected == {str(path): digest(path) for path in split.rglob("*.h5")}
            if args.clipped or args.defaults:
                run(label + "_reject_changed_window", twist, 2, checkpoint, clip_width=4,
                    rejection="Training checkpoint stage identity or configuration mismatch")
                assert protected == {str(path): digest(path) for path in split.rglob("*.h5")}
            if args.defaults:
                changed = "jaqmc" if args.defaults != "jaqmc" else "deepsolid"
                run(label + "_reject_changed_defaults", twist, 2, checkpoint, defaults=changed,
                    rejection="Training checkpoint stage identity or configuration mismatch")
                assert protected == {str(path): digest(path) for path in split.rglob("*.h5")}
                if label == twists[0][0]:
                    run(label + "_explicit_override", twist, 1, rate_override="0.007")
            report["cases"].append(dict(twist=twist, exact_model_optimizer_sampler_restart=True,
                                         imaginary_head_changed=not args.real_limit,
                                         real_limit=args.real_limit, samples_per_update=2 * args.ranks))
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
