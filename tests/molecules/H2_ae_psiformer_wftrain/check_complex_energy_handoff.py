#!/usr/bin/env python3
"""Prepare/validate one CPU-complex MSE-to-ENERGY list; never launch a job."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from check_complex_cpu_ensemble import digest, save, validate_slot, scalar_rows, datasets, compare_archives, text_value


def prepare(root, baseline, runtime_baseline):
    """Append one ordinary ENERGY stage without changing the frozen MSE controls."""
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((baseline / "manifest.json").read_text())
    for slot in (0, 1):
        tree = ET.parse(baseline / "mixed" / f"slot{slot}.xml")
        energy = copy.deepcopy(tree.find("qmc"))
        training = energy.find("training")
        training.set("objective", "energy")
        training.find("parameter[@name='iterations']").text = "1"
        for name in ("teacher", "target_provenance", "orbital_target"):
            training.remove(training.find(f"parameter[@name='{name}']"))
        tree.getroot().append(energy)
        tree.write(root / f"slot{slot}.xml", encoding="utf-8", xml_declaration=True)
    (root / "inputs.txt").write_text("slot0.xml\nslot1.xml\n")
    manifest.update(status="prepared_requires_review", baseline=str(runtime_baseline),
                    baseline_manifest_sha256=digest(baseline / "manifest.json"),
                    handoff_helper_sha256=digest(__file__), cases=["handoff"], per_launch_seconds=60,
                    inputs={name: digest(root / name) for name in ("slot0.xml", "slot1.xml", "inputs.txt")})
    save(root / "manifest.json", manifest)


def validate(root):
    """Require exact MSE preservation followed by a real version-2-to-3 ENERGY update."""
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "ready_to_execute" and digest(__file__) == manifest["handoff_helper_sha256"]
    baseline = Path(manifest["baseline"])
    assert digest(baseline / "manifest.json") == manifest["baseline_manifest_sha256"]
    for line in (baseline / "REMOTE_HDF_SHA256SUMS").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        assert digest(baseline / relative) == checksum, relative
    for slot in (0, 1):
        assert validate_slot(root, slot) == validate_slot(baseline / "mixed", slot)
        stem = f"slot{slot}.g{slot:03d}"
        for pattern in ("*.parameters.h5", "*.state.h5", "*.rank_*.h5"):
            compare_archives(root / f"{stem}.s000.training", baseline / "mixed" / f"{stem}.s000.training", pattern)
        with (root / f"{stem}.s001.training_info.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        row = rows[0]
        assert (row["objective"], row["optimizer"], row["committed"], row["samples"]) == ("energy", "adam", "1", "2")
        assert (row["sample_version"], row["parameter_version"]) == ("2", "3")
        assert all(math.isfinite(float(row[key])) for key in ("mean_energy", "variance", "gradient_norm"))
        assert float(row["gradient_norm"]) > 0
        previous = sorted((root / f"{stem}.s000.training").glob("*.parameters.h5"))[-1]
        models = sorted((root / f"{stem}.s001.training").glob("*.parameters.h5"))
        assert len(models) == 1
        values = datasets(models[0])["values"]
        assert np.isfinite(values).all() and np.any(values != datasets(previous)["values"])
        energy_bundle = models[0].parent / (models[0].name.removesuffix(".parameters.h5") + ".checkpoint")
        mse_bundle = previous.parent / (previous.name.removesuffix(".parameters.h5") + ".checkpoint")
        assert len(list(energy_bundle.glob("*.rank_*.h5"))) == 2
        assert len(list(energy_bundle.glob("*.manifest.h5"))) == len(list(energy_bundle.glob("*.state.h5"))) == 1
        energy_state = datasets(next(energy_bundle.glob("*.state.h5")))
        mse_state = datasets(next(mse_bundle.glob("*.state.h5")))
        stage = "wftrain_checkpoint/stage/"
        assert text_value(mse_state[stage + "stage_id"]) == "psiformer/orbital-pretraining"
        assert text_value(energy_state[stage + "stage_id"]) == "all_electron_energy"
        assert text_value(mse_state[stage + "configuration_fingerprint"]) != text_value(energy_state[stage + "configuration_fingerprint"])
        scalars = sorted((root / f"{stem}.s001.inference").glob("*.scalar.dat"))
        assert len(scalars) == 1
        scalar_rows(scalars[0])
    save(root / "validation.json", dict(status="passed_exact_MSE_then_energy_handoff", groups=2, energy_source_version=2, energy_accepted_version=3))


def main():
    """Keep local input generation and post-run validation separate from scheduling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--runtime-baseline", type=Path)
    args = parser.parse_args()
    if args.baseline:
        assert args.runtime_baseline
        prepare(args.root, args.baseline, args.runtime_baseline)
    else:
        try:
            validate(args.root)
        except BaseException as error:
            save(args.root / "validation.json", dict(status="failed", error=str(error)))
            raise


if __name__ == "__main__":
    main()
