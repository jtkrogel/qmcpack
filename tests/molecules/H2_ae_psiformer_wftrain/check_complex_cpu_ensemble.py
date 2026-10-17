#!/usr/bin/env python3
"""Three finite CPU-complex XML-list launches with matched WORLD-slot controls."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "H2_ae_conventional_wftrain"))
from check_cpu_ensemble import digest, save
from check_sj_training import compare_archives, datasets, text_value


def dependencies():
    """Pin both reused helpers rather than silently changing validation code."""
    base = Path(__file__).resolve().parents[1] / "H2_ae_conventional_wftrain"
    return {name: digest(base / name) for name in ("check_cpu_ensemble.py", "check_sj_training.py")}


def prepare(root, application, application_sha, source, library_path):
    """Generate inert, matched-slot inputs; a reviewed CPU artifact is still required."""
    root.mkdir(parents=True, exist_ok=False)
    fixture = Path(__file__).with_name("wftrain_complex_orbital_mse.xml")
    hashes = {}
    for label, models in (("mixed", ("A", "B")), ("a_pair", ("A", "A")), ("b_pair", ("B", "B"))):
        directory = root / label
        directory.mkdir()
        for slot, model in enumerate(models):
            tree = ET.parse(fixture)
            tree.find("project").set("id", f"slot{slot}")
            tree.find("random").set("seed", str((271, 811)[slot]))
            if model == "B":
                tree.find(".//psiformer").set("initialization_seed", "23")
                for node in (tree.find(".//psiformer"), tree.find(".//sposet")):
                    node.set("twist", "-0.09 0.13 0.04")
            tree.write(directory / f"slot{slot}.xml", encoding="utf-8", xml_declaration=True)
        (directory / "inputs.txt").write_text("slot0.xml\nslot1.xml\n")
        hashes.update({str(path.relative_to(root)): digest(path) for path in directory.iterdir()})
    save(root / "manifest.json", dict(status="prepared_requires_pin_review", application=str(application),
         application_sha256=application_sha, application_source=source, runtime_library_path=library_path,
         controller_sha256=digest(__file__), dependencies=dependencies(), fixture_sha256=digest(fixture),
         inputs=hashes, cases=["mixed", "a_pair", "b_pair"], ranks=4, groups=2, ranks_per_group=2,
         threads_per_rank=1, per_launch_seconds=35, aggregate_seconds=120, retries=0))


def scalar_rows(path):
    """Require a complete two-rank inference result, omitting only CPU timing."""
    lines = path.read_text().splitlines()
    assert any("complete yes" in line for line in lines), path
    header = next(line.split()[1:] for line in lines if line.startswith("#") and "BlockWeight" in line)
    data = [line.split() for line in lines if line.strip() and not line.startswith("#")]
    assert len(data) == 1 and len(data[0]) == len(header), path
    row = dict(zip(header, data[0]))
    assert all(math.isfinite(float(value)) for value in row.values()), path
    assert float(row["BlockWeight"]) == 2 and 0 < float(row["AcceptRatio"]) <= 1, path
    return header, {key: value for key, value in row.items() if key != "BlockCPU"}


def validate_slot(directory, slot):
    """Check actual group counts, finite changing paired state and complete bundles."""
    stem = f"slot{slot}.g{slot:03d}.s000"
    with (directory / f"{stem}.training_info.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    models = sorted((directory / f"{stem}.training").glob("*.parameters.h5"))
    assert len(rows) == len(models) == 2
    payloads, moments = [], []
    for iteration, (row, model) in enumerate(zip(rows, models)):
        assert int(row["iteration"]) == iteration and row["committed"] == "1" and row["samples"] == "2", row
        assert row["objective"] == "orbital_mse" and row["optimizer"] == "adam", row
        assert int(row["parameter_version"]) == iteration + 1 and int(row["sample_version"]) == iteration
        assert all(math.isfinite(float(row[key])) for key in ("orbital_loss", "gradient_norm"))
        assert float(row["orbital_loss"]) >= 0 and float(row["gradient_norm"]) > 0
        payloads.append(datasets(model))
        assert np.isfinite(payloads[-1]["values"]).all()
        bundle = model.parent / (model.name.removesuffix(".parameters.h5") + ".checkpoint")
        assert len(list(bundle.glob("*.rank_*.h5"))) == 2
        states = list(bundle.glob("*.state.h5"))
        assert len(states) == len(list(bundle.glob("*.manifest.h5"))) == 1
        state = datasets(states[0])
        moments.append([state[f"wftrain_checkpoint/optimizer/{name}_moment"] for name in ("first", "second")])
        assert all(np.isfinite(value).all() and value.size == payloads[-1]["values"].size for value in moments[-1])
    assert np.any(payloads[0]["values"] != payloads[1]["values"])
    names = [text_value(value) for value in np.asarray(payloads[1]["layout/names"]).reshape(-1)]
    offsets = np.asarray(payloads[1]["layout/offsets"]).reshape(-1)
    imaginary = [slice(int(offsets[i]), int(offsets[i + 1])) for i, name in enumerate(names) if name == "w_imag"]
    assert imaginary and any(np.any(payloads[0]["values"][part] != payloads[1]["values"][part]) for part in imaginary)
    assert all(np.any(first != second) for first, second in zip(*moments))
    assert (directory / f"{stem}.last_checkpoint.h5").is_file()
    scalars = sorted((directory / f"{stem}.inference").glob("*.scalar.dat"))
    assert len(scalars) == 2
    return ([{key: value for key, value in row.items() if key not in
              ("sampling_seconds", "total_seconds", "resident_bytes")} for row in rows],
            [scalar_rows(path) for path in scalars])


def execute(root, allocation):
    """Run each approved list once; terminate owned groups on signals or deadlines."""
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "ready_to_execute" and manifest["controller_sha256"] == digest(__file__)
    assert manifest["dependencies"] == dependencies()
    assert digest(manifest["application"]) == manifest["application_sha256"]
    for name, expected in manifest["inputs"].items():
        assert digest(root / name) == expected, name
    assert subprocess.check_output(["squeue", "-j", allocation, "-h", "-o", "%T %N"], text=True).strip() == "RUNNING or-condo-c[555-556]"
    started, active = time.monotonic(), None
    result = dict(status="running", cases=[])
    marker = root / "execution.json"
    with marker.open("x") as stream:
        json.dump(result, stream)

    def stop(signum=None, frame=None):
        """Let srun cancel its step before forcing termination of the owned group."""
        if active is not None and active.poll() is None:
            try:
                os.killpg(active.pid, signal.SIGTERM)
                active.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if active.poll() is None:
                    try:
                        os.killpg(active.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    active.wait()
        if signum is not None:
            raise SystemExit(f"ensemble interrupted by signal {signum}")

    previous = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
    environment = dict(os.environ, OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1")
    environment["LD_LIBRARY_PATH"] = manifest["runtime_library_path"] + ":" + environment.get("LD_LIBRARY_PATH", "")
    try:
        for label in manifest["cases"]:
            assert 120 - (time.monotonic() - started) >= 35, "insufficient aggregate budget; no retry"
            directory = root / label
            command = ["srun", "--jobid=" + allocation, "--exclusive", "--exact", "--nodes=2",
                       "--nodelist=or-condo-c[555-556]", "--ntasks=4", "--ntasks-per-node=2", "--cpus-per-task=1",
                       "--distribution=cyclic:cyclic", "--cpu-bind=cores", "--mem=14G", "--time=00:01:00",
                       "--kill-on-bad-exit=1", sys.executable, str(Path(__file__).resolve()), "--rank",
                       manifest["application"], "inputs.txt"]
            save(directory / "launch.json", dict(command=command, environment={key: environment[key] for key in
                 ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")}))
            case_start = time.monotonic()
            with (directory / "console.log").open("w") as output:
                active = subprocess.Popen(command, cwd=directory, env=environment, stdout=output,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                assert active.wait(timeout=35) == 0, label
            placements = {int(line.split()[1]): line.split()[2] for line in
                          (directory / "console.log").read_text().splitlines() if line.startswith("ENSEMBLE_RANK ")}
            assert len(placements) == 4
            assert all(len({placements[2 * group], placements[2 * group + 1]}) == 2 for group in (0, 1))
            for slot in (0, 1):
                validate_slot(directory, slot)
            result["cases"].append(dict(case=label, elapsed=time.monotonic() - case_start, placements=placements))
            save(marker, result)
        for slot, peer in ((0, "a_pair"), (1, "b_pair")):
            assert validate_slot(root / "mixed", slot) == validate_slot(root / peer, slot)
            stem = f"slot{slot}.g{slot:03d}.s000.training"
            for pattern in ("*.parameters.h5", "*.state.h5", "*.rank_*.h5"):
                compare_archives(root / "mixed" / stem, root / peer / stem, pattern)
        assert time.monotonic() - started <= 120, "aggregate validation deadline"
        result["status"] = "passed_exact_peer_isolation"
    except BaseException as error:
        stop()
        result.update(status="failed", error=str(error))
        raise
    finally:
        result["elapsed"] = time.monotonic() - started
        save(marker, result)
        for number, handler in previous.items():
            signal.signal(number, handler)
        print(json.dumps(result), flush=True)


def main():
    """Separate rank reporting, inert preparation and explicitly reviewed execution."""
    if len(sys.argv) > 1 and sys.argv[1] == "--rank":
        print("ENSEMBLE_RANK", os.environ["SLURM_PROCID"], socket.gethostname(), flush=True)
        os.execv(sys.argv[2], sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    for name in ("application", "application-sha256", "application-source", "runtime-library-path"):
        parser.add_argument("--" + name)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allocation", default="5752988")
    args = parser.parse_args()
    if args.execute:
        execute(args.root.resolve(), args.allocation)
    else:
        assert all((args.application, args.application_sha256, args.application_source, args.runtime_library_path))
        prepare(args.root.resolve(), args.application, args.application_sha256, args.application_source, args.runtime_library_path)


if __name__ == "__main__":
    main()
