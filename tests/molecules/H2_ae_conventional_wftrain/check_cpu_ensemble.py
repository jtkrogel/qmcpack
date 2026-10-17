#!/usr/bin/env python3
"""Three bounded native CPU XML-list launches; no production scheduler changes."""
import argparse
import csv
import hashlib
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
from check_sj_training import compare_archives, datasets, text_value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(root, source, executable, application_sha):
    root.mkdir(parents=True, exist_ok=False)
    fixtures = {"sj": source / "tests/molecules/H2_ae_conventional_wftrain/wftrain_sj.xml",
                "mdj": source / "tests/molecules/H4_ae_conventional_wftrain/wftrain_mdj.xml"}
    hashes = {}
    for label, models in (("mixed", ("sj", "mdj")), ("sj_pair", ("sj", "sj")),
                          ("mdj_pair", ("mdj", "mdj"))):
        directory = root / label
        directory.mkdir()
        for slot, model in enumerate(models):
            tree = ET.parse(fixtures[model])
            tree.find("project").set("id", f"slot{slot}")
            tree.find("random").set("seed", str((271, 811)[slot]))
            tree.find("qmc/training/parameter[@name='learning_rate']").text = ("0.0001", "0.0002")[slot]
            path = directory / f"slot{slot}.xml"
            tree.write(path, encoding="utf-8", xml_declaration=True)
            hashes[str(path.relative_to(root))] = digest(path)
        (directory / "inputs.txt").write_text("slot0.xml\nslot1.xml\n")
        hashes[f"{label}/inputs.txt"] = digest(directory / "inputs.txt")
    save(root / "manifest.json", dict(application=str(executable), application_sha256=application_sha,
         application_source="5b52f17e854a111dbae0f2783285f054a351f9bb", controller_sha256=digest(__file__),
         shared_checker_sha256=digest(Path(__file__).with_name("check_sj_training.py")),
         fixture_sha256={key: digest(value) for key, value in fixtures.items()}, inputs=hashes,
         cases=["mixed", "sj_pair", "mdj_pair"], ranks=4, groups=2, ranks_per_group=2,
         threads_per_rank=2, nodes=["or-condo-c555", "or-condo-c556"], per_launch_seconds=35,
         aggregate_seconds=120, retries=0,
         runtime_library_path="/sw/cades-besms/spack-envs/opt/base/oneapi-2024.1.0/hdf5-1.14.3-jhkpwiobpag5ywos5jj46kgc2p6gmvqo/lib"))


def scalar_rows(path):
    lines = path.read_text().splitlines()
    assert any("complete yes" in line for line in lines), path
    header = next(line.split()[1:] for line in lines if line.startswith("#") and "BlockWeight" in line)
    data = [line.split() for line in lines if line.strip() and not line.startswith("#")]
    assert len(data) == 1 and len(data[0]) == len(header), path
    row = dict(zip(header, map(float, data[0])))
    assert all(math.isfinite(value) for value in row.values()), path
    assert row["BlockWeight"] == 4 and 0 < row["AcceptRatio"] <= 1, path
    return {key: value for key, value in row.items() if key != "BlockCPU"}


def validate_slot(directory, slot):
    root = f"slot{slot}.g{slot:03d}.s000"
    tree = ET.parse(directory / f"slot{slot}.xml")
    with (directory / f"{root}.training_info.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    initial = [float(node.get("coeff")) for node in tree.findall(".//detlist/ci")[1:]]
    initial += [float(value) for node in tree.findall(".//jastrow/correlation/coefficients")
                for value in node.text.split()]
    previous = np.asarray(initial)
    models = sorted((directory / f"{root}.training").glob("*.parameters.h5"))
    assert len(models) == 2
    scientific = []
    for iteration, (row, model) in enumerate(zip(rows, models)):
        assert int(row["iteration"]) == iteration and row["committed"] == "1"
        assert int(row["samples"]) == 8
        assert int(row["parameter_version"]) == iteration + 1 and int(row["sample_version"]) == iteration
        assert all(math.isfinite(float(row[key])) for key in ("mean_energy", "variance", "gradient_norm"))
        assert float(row["gradient_norm"]) > 0
        payload = datasets(model)
        parameters = payload["conventional_wavefunction/parameters"].reshape(-1)
        assert parameters.size == len(initial) and np.isfinite(parameters).all()
        assert not np.array_equal(previous, parameters)
        previous = parameters
        # Exported original XML must preserve the fixed SPO and fixed CI0.
        saved = ET.fromstring(text_value(payload["conventional_wavefunction/wavefunction_xml"]))
        fixed = tree.find(".//sposet/coefficient")
        np.testing.assert_array_equal(np.fromstring(saved.find(".//sposet/coefficient").text, sep=" "),
                                      np.fromstring(fixed.text, sep=" "))
        ci = saved.findall(".//detlist/ci")
        if ci:
            assert ci[0].get("coeff") == tree.find(".//detlist/ci").get("coeff")
        bundle = model.parent / (model.name.removesuffix(".parameters.h5") + ".checkpoint")
        assert len(list(bundle.glob("*.rank_*.h5"))) == 2
        assert len(list(bundle.glob("*.state.h5"))) == len(list(bundle.glob("*.manifest.h5"))) == 1
        scientific.append({key: value for key, value in row.items()
                           if key not in ("sampling_seconds", "total_seconds", "resident_bytes")})
    assert (directory / f"{root}.last_checkpoint.h5").is_file()
    inference = sorted((directory / f"{root}.inference").glob("*.scalar.dat"))
    assert len(inference) == 2
    return dict(history=scientific, inference=[scalar_rows(path) for path in inference], parameters=len(initial))


def compare_slot(left, right, slot):
    a, b = validate_slot(left, slot), validate_slot(right, slot)
    assert a == b, (left, right, slot, "scientific history/inference mismatch")
    family = f"slot{slot}.g{slot:03d}.s000.training"
    for pattern in ("*.parameters.h5", "*.state.h5", "*.rank_*.h5"):
        compare_archives(left / family, right / family, pattern)
    return a


def execute(root, allocation):
    manifest = json.loads((root / "manifest.json").read_text())
    assert digest(__file__) == manifest["controller_sha256"]
    assert digest(Path(__file__).with_name("check_sj_training.py")) == manifest["shared_checker_sha256"]
    assert digest(manifest["application"]) == manifest["application_sha256"]
    for name, expected in manifest["inputs"].items():
        assert digest(root / name) == expected, name
    assert subprocess.check_output(["squeue", "-j", allocation, "-h", "-o", "%T %N"], text=True).strip() == "RUNNING or-condo-c[555-556]"
    started = time.monotonic()
    result = dict(status="running", cases=[])
    marker = root / "execution.json"
    with marker.open("x") as handle:
        json.dump(result, handle)
    environment = dict(os.environ, OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1")
    environment["LD_LIBRARY_PATH"] = manifest["runtime_library_path"] + ":" + environment.get("LD_LIBRARY_PATH", "")
    for label in manifest["cases"]:
        directory = root / label
        command = ["srun", "--jobid=" + allocation, "--exclusive", "--exact", "--nodes=2",
                   "--nodelist=or-condo-c[555-556]", "--ntasks=4", "--ntasks-per-node=2", "--cpus-per-task=2",
                   "--distribution=cyclic:cyclic", "--cpu-bind=cores", "--mem=14G", "--time=00:01:00",
                   "--kill-on-bad-exit=1", sys.executable, str(Path(__file__).resolve()), "--rank",
                   manifest["application"], "inputs.txt"]
        save(directory / "launch.json", dict(command=command, environment={key: environment[key] for key in
             ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")}))
        case_start = time.monotonic()
        try:
            remaining = 120 - (case_start - started)
            assert remaining >= 35, "aggregate deadline leaves insufficient launch budget"
            with (directory / "console.log").open("w") as output:
                process = subprocess.Popen(command, cwd=directory, env=environment, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=3)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=2)
                    raise RuntimeError("35 second launch deadline; no retry")
            assert code == 0, (label, code)
            placements = {}
            for line in (directory / "console.log").read_text().splitlines():
                if line.startswith("ENSEMBLE_RANK "):
                    _, rank, host = line.split()
                    placements[int(rank)] = host
            assert len(placements) == 4
            assert all(len({placements[rank] for rank in (2 * group, 2 * group + 1)}) == 2 for group in (0, 1))
            slots = [validate_slot(directory, slot) for slot in (0, 1)]
            result["cases"].append(dict(case=label, elapsed=time.monotonic() - case_start, placements=placements,
                                        parameter_counts=[value["parameters"] for value in slots]))
        except Exception as error:
            result.update(status="failed", error=str(error), elapsed=time.monotonic() - started)
            save(marker, result)
            raise
        save(marker, result)
    try:
        compare_slot(root / "mixed", root / "sj_pair", 0)
        compare_slot(root / "mixed", root / "mdj_pair", 1)
    except Exception as error:
        result.update(status="failed_exact_peer_comparison", error=str(error), elapsed=time.monotonic() - started)
        save(marker, result)
        raise
    result.update(status="passed_exact_peer_isolation", elapsed=time.monotonic() - started)
    save(marker, result)
    print(json.dumps(result), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--rank":
        print("ENSEMBLE_RANK", os.environ["SLURM_PROCID"], socket.gethostname(), flush=True)
        os.execv(sys.argv[2], sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--application", type=Path)
    parser.add_argument("--application-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allocation", default="5752988")
    args = parser.parse_args()
    if args.execute:
        execute(args.root.resolve(), args.allocation)
    else:
        assert args.source and args.application and args.application_sha256
        prepare(args.root.resolve(), args.source.resolve(), args.application, args.application_sha256)
        print("Prepared only; no execution requested", flush=True)


if __name__ == "__main__":
    main()
