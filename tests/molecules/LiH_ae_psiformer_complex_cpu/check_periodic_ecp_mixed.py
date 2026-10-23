#!/usr/bin/env python3
"""Qualify admitted real Gamma scalar-ECP mixed inference at matched configurations."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from check_modern_integration import parameter, sha256_file


def coordinates(path):
    try:
        import h5py
    except ImportError:
        dump = subprocess.check_output(["h5dump", "-y", "-w", "0", "-m", "%.17g",
                                        "-d", "/state_0/walkers", str(path)], universal_newlines=True)
        data = re.search(r"DATA\s*\{([^}]+)\}", dump)
        if not data:
            raise RuntimeError("Missing native walker coordinate checkpoint")
        return [float(value.strip()) for value in data.group(1).split(",")]
    else:
        with h5py.File(path, "r") as archive:
            return archive["state_0/walkers"][()].reshape(-1).tolist()


def scalars(path):
    lines = path.read_text().splitlines()
    header = lines[0].lstrip("#").split()
    rows = [[float(token) for token in line.split()] for line in lines[1:] if line.strip()]
    if len(rows) != 2 or any(len(row) != len(header) for row in rows):
        raise RuntimeError("Expected two complete physical scalar blocks")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise RuntimeError("Nonfinite complete physical observation")
    if not any("nonlocal" in name.lower() for name in header):
        raise RuntimeError("Missing native nonlocal pseudopotential scalar")
    if any(row[header.index("AcceptRatio")] != 1 for row in rows):
        raise RuntimeError("Matched-configuration probe requires unit joint acceptance")
    keep = [i for i, name in enumerate(header) if name not in ("BlockCPU", "BlockTime", "Time")]
    return header, [[row[i] for i in keep] for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--real-profile-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-zero-retries", action="store_true")
    parser.add_argument("--require-final-counters", action="store_true")
    parser.add_argument("--gpu", choices=("cuda", "hip"), default="cuda")
    args = parser.parse_args()
    binary = args.binary.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = ET.parse(args.fixture)
    physical = original.getroot().find("qmcsystem")
    cell = physical.find("simulationcell")
    if cell is None or cell.find("parameter[@name='bconds']").text.split() != ["p"] * 3:
        raise RuntimeError("Qualification requires an actual bulk periodic fixture")
    pseudo_nodes = physical.findall("./hamiltonian/pairpot[@type='pseudo']")
    if not pseudo_nodes or any(node.get("DLA", "no").lower() != "no" for node in pseudo_nodes):
        raise RuntimeError("Qualification requires ordinary scalar pseudopotential locality")
    profile = ET.parse(args.real_profile_fixture).getroot().find(".//wavefunction/psiformer")
    observations, accepted_coordinates = {}, {}
    receipt = {"binary_sha256": sha256_file(binary), "physical_fixture_sha256": sha256_file(args.fixture),
               "real_profile_sha256": sha256_file(args.real_profile_fixture), "runs": {}}
    for backend in ("cpu", args.gpu):
        directory = output / backend
        directory.mkdir()
        tree = copy.deepcopy(original)
        root = tree.getroot()
        root.find("project").set("id", "mixed_ecp")
        system = root.find("qmcsystem")
        wavefunction = system.find("wavefunction")
        previous = wavefunction.find("psiformer")
        head = copy.deepcopy(profile)
        head.set("system", "pseudopotential")
        head.set("source", previous.get("source"))
        head.set("initialization_seed", previous.get("initialization_seed"))
        head.set("gpu", backend)
        head.set("optimize", "no")
        head.set("precision", "fp64_reference" if backend == "cpu" else "fp32_dense_fp64_islands")
        wavefunction.remove(previous)
        wavefunction.append(head)
        for pseudo in system.findall("./hamiltonian/pairpot/pseudo"):
            pseudo.set("href", str((args.fixture.resolve().parent / pseudo.get("href")).resolve()))
        for node in list(root.findall("qmc")):
            root.remove(node)
        qmc = ET.SubElement(root, "qmc", method="vmc_batch", move="alle", checkpoint="0")
        for name, value in {"walkers_per_rank": 2, "crowds": 1, "blocks": 2, "steps": 2,
                            "subSteps": 1, "warmupSteps": 0, "timestep": "1e-16", "usedrift": "no"}.items():
            parameter(qmc, name, value)
        source = directory / "input.xml"
        tree.write(source, encoding="unicode")
        receipt["runs"][backend] = hashlib.sha256(source.read_bytes()).hexdigest()
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        with (directory / "qmcpack.log").open("w") as log:
            result = subprocess.run([str(binary), source.name], cwd=directory, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=600)
        if result.returncode:
            raise RuntimeError(f"{backend} application exit {result.returncode}; inspect {directory}")
        console = (directory / "qmcpack.log").read_text()
        if backend == args.gpu:
            markers = re.findall(r"mixed inference completed: mixed_batches=(\d+), whole_fp64_retries=(\d+), retry_reason=(.*)", console)
            if not markers or not any(int(mixed) > 0 for mixed, retries, reason in markers):
                raise RuntimeError("No proved actual mixed execution")
            if args.require_zero_retries and any(int(retries) for mixed, retries, reason in markers):
                raise RuntimeError("Strict zero-retry mixed qualification encountered whole-configuration FP64 replay")
            previous = 0
            events = []
            for mixed, retries, reason in markers:
                count = int(retries)
                if count > previous:
                    if not reason.strip():
                        raise RuntimeError("Whole-configuration FP64 replay omitted its reason")
                    events.append({"new_replays": count - previous, "reason": reason})
                previous = count
            mixed, retries = (int(value) for value in markers[-1][:2])
            receipt["precision"] = {
                "requested": "fp32_dense_fp64_islands", "effective": "mixed affine graphs with complete FP64 replay",
                "counts_scope": "last emitted diagnostic; successful mixed count is a lower bound, not a final total",
                "observed_successful_mixed_graphs": mixed, "observed_complete_fp64_replays": retries,
                "observed_mixed_fraction": mixed / (mixed + retries), "fallback_events": events,
                "retry_contract": "same complete requested graph evaluated again at unchanged positions; no field splice"}
            summaries = re.findall(r"mixed inference summary: scope=all_destroyed_contexts, contexts=(\d+), mixed_batches=(\d+), whole_fp64_retries=(\d+)", console)
            if args.require_final_counters and not summaries:
                raise RuntimeError("Missing final complete-context precision counters")
            if summaries:
                contexts = sum(int(contexts) for contexts, mixed, retries in summaries)
                successes = sum(int(mixed) for contexts, mixed, retries in summaries)
                replays = sum(int(retries) for contexts, mixed, retries in summaries)
                if contexts == 0 or successes == 0:
                    raise RuntimeError("Final counters prove no successful mixed execution")
                if args.require_zero_retries and replays:
                    raise RuntimeError("Strict zero-retry mode found FP64 replay in final counters")
                receipt["precision"]["final_totals"] = {
                    "scope": "all_destroyed_contexts", "contexts": contexts,
                    "successful_mixed_graphs": successes, "complete_fp64_replays": replays,
                    "mixed_fraction": successes / (successes + replays)}
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if f"numerical device tile completed: backend={args.gpu}" not in console:
                raise RuntimeError("Missing actual completed device numerical tile")
        observations[backend] = scalars(directory / "mixed_ecp.s000.scalar.dat")
        accepted_coordinates[backend] = coordinates(directory / "mixed_ecp.s000.config.h5")
    if accepted_coordinates["cpu"] != accepted_coordinates[args.gpu]:
        raise RuntimeError("CPU FP64 and GPU mixed accepted coordinates differ")
    if observations["cpu"][0] != observations[args.gpu][0]:
        raise RuntimeError("Physical scalar schemas differ")
    maximum = 0.0
    for reference, mixed in zip(observations["cpu"][1], observations[args.gpu][1]):
        for left, right in zip(reference, mixed):
            if not math.isclose(left, right, rel_tol=5e-4, abs_tol=5e-4):
                raise RuntimeError(f"Mixed physical observable mismatch: {left} versus {right}")
            maximum = max(maximum, abs(left - right))
    receipt["comparison"] = {"exact_accepted_coordinates": True, "all_joint_proposals_accepted": True,
                             "maximum_physical_scalar_difference": maximum,
                             "absolute_tolerance": 5e-4, "relative_tolerance": 5e-4,
                             "complete_scalar_schema": observations["cpu"][0]}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"PASS periodic scalar-ECP mixed inference: exact accepted coordinates, native nonlocal physics, maximum difference {maximum:.12g}; precision counts/reasons in receipt.json")


if __name__ == "__main__":
    main()
