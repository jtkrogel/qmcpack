#!/usr/bin/env python3
"""Exercise modern estimator observation/resumed sampling or PBYP DMC from XML."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def parameter(node, name, value):
    child = ET.SubElement(node, "parameter", name=name)
    child.text = str(value)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_estimators(path):
    datasets = ("/nofk/value", "/rdm/number_matrix/u/value", "/rdm/number_matrix/d/value")
    try:
        import h5py
    except ImportError:
        for dataset in datasets:
            dump = subprocess.check_output(["h5dump", "-y", "-w", "0", "-d", dataset, str(path)], universal_newlines=True)
            data = re.search(r"DATA\s*\{([^}]+)\}", dump)
            if not data:
                raise RuntimeError(f"Missing estimator dataset {dataset}")
            values = [float(value.strip()) for value in data.group(1).split(",")]
            if not values or not all(math.isfinite(value) for value in values) or not any(value != 0 for value in values):
                raise RuntimeError(f"Nonfinite or trivial estimator dataset {dataset}")
    else:
        import numpy as np
        with h5py.File(path, "r") as stat:
            for dataset in datasets:
                values = stat[dataset][()]
                if not values.size or not np.isfinite(values).all() or not np.any(values != 0):
                    raise RuntimeError(f"Nonfinite or trivial estimator dataset {dataset}")


def prepare(fixture, mode, gpu, evaluator=None, timestep="1e-5", steps=2):
    tree = ET.parse(fixture)
    simulation = tree.getroot()
    simulation.find("project").set("id", "modern_probe")
    system = simulation.find("qmcsystem")
    for pseudo in system.findall("./hamiltonian/pairpot/pseudo"):
        href = Path(pseudo.get("href"))
        if not href.is_absolute():
            pseudo.set("href", str((fixture.resolve().parent / href).resolve()))
    wavefunction = system.find("wavefunction")
    wavefunction.find("psiformer").set("gpu", gpu)
    if gpu in ("cuda", "hip") and "complex" in wavefunction.find("psiformer").get("initialization", ""):
        wavefunction.find("psiformer").set("max_gpu_numeric_bytes", str(64 * 1024**3))
        wavefunction.find("psiformer").attrib.pop("max_cpu_shared_numeric_bytes", None)
    for node in list(simulation.findall("qmc")):
        simulation.remove(node)
    if mode == "estimators":
        # This projection basis is independent of the many-electron ansatz.
        collection = ET.Element("sposet_collection", type="heg")
        ET.SubElement(collection, "sposet", type="heg", name="projection", size="3")
        wavefunction.insert(0, collection)
        estimators = ET.SubElement(system, "estimators")
        ET.SubElement(estimators, "estimator", type="MomentumDistribution",
                      name="momentum", samples="2", kmax="1")
        density = ET.SubElement(estimators, "estimator", type="OneBodyDensityMatrices", name="rdm")
        if evaluator:
            parameter(density, "evaluator", evaluator)
        for key, value in {"basis": "projection", "integrator": "uniform_grid",
                           "points": 2, "normalized": "yes"}.items():
            parameter(density, key, value)
    for section in range(2):
        qmc = ET.SubElement(simulation, "qmc", method="dmc_batch" if mode == "dmc" else "vmc_batch",
                            move="n_electron" if mode == "selected" else "pbyp", checkpoint="0")
        settings = {"walkers_per_rank": 6, "crowds": 2, "blocks": 2,
                    "steps": steps, "warmupSteps": 0, "timestep": timestep,
                    "blocks_between_recompute": 1}
        if mode != "dmc":
            settings.update(usedrift="no", subSteps=1)
            if mode == "selected":
                settings.update(electrons_per_move=1, electron_selection="cyclic")
        else:
            settings.update(nonlocalmoves="no", reconfiguration="no", reserve=2)
            settings.update(debug_checks="checkGL_after_moves")
        for key, value in settings.items():
            parameter(qmc, key, value)
    return tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--mode", choices=("estimators", "dmc", "selected"), required=True)
    parser.add_argument("--gpu", choices=("cpu", "cuda", "hip"), default="cpu")
    parser.add_argument("--evaluator", choices=("matrix", "loop"))
    parser.add_argument("--timestep", default="1e-5")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--mpi-launcher", choices=("mpiexec", "srun"), default="mpiexec")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--expect-tmove-rejection", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.expect_tmove_rejection and args.mode != "dmc":
        parser.error("T-move admission rejection applies only to DMC")
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run_", dir=output_root))
    source = output / "input.xml"
    tree = prepare(args.fixture, args.mode, args.gpu, args.evaluator, args.timestep, args.steps)
    if args.expect_tmove_rejection:
        for node in tree.getroot().findall("./qmc/parameter[@name='nonlocalmoves']"):
            node.text = "v1"
    tree.write(source, encoding="unicode")
    binary = args.binary.resolve()
    receipt = {"binary_sha256": sha256_file(binary),
               "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "mode": args.mode,
               "mpi_ranks": args.mpi_ranks,
               "environment": {name: os.environ.get(name) for name in (
                   "PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "PSIFORMER_GPU_ACTIVE_INFERENCE_CROWDS",
                   "PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES")}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (output / "qmcpack.log").open("w") as log:
        command = [str(binary), source.name]
        # Keep an expected MPI_Abort inside its own Slurm child step so it cannot
        # terminate the surrounding Python/CTest step before rejection is checked.
        isolate_rejection = args.expect_tmove_rejection and "SLURMD_NODENAME" in os.environ
        if args.mpi_ranks > 1 or args.mpi_launcher == "srun" or isolate_rejection:
            if args.mpi_launcher == "srun" or isolate_rejection:
                command = ["srun", "--overlap", "--cpu-bind=none", "--nodes=1", "--ntasks", str(args.mpi_ranks),
                           "--cpus-per-task=2", "--nodelist", os.environ["SLURMD_NODENAME"]] + command
            else:
                command = ["mpiexec", "-n", str(args.mpi_ranks)] + command
        result = subprocess.run(command, cwd=output, stdout=log,
                                stderr=subprocess.STDOUT, timeout=600)
    if args.expect_tmove_rejection:
        message = "PsiFormer batched DMC requires nonlocalmoves=no; T-moves are not supported"
        if result.returncode == 0 or message not in (output / "qmcpack.log").read_text():
            raise RuntimeError("Expected explicit unsupported T-move admission rejection")
        if list(output.glob("*.scalar.dat")):
            raise RuntimeError("T-move rejection occurred after production output creation")
        print(f"PASS explicit T-move rejection before production: {output}")
        return
    if result.returncode:
        raise RuntimeError(f"Application exit {result.returncode}; inspect {output / 'qmcpack.log'}")
    if args.restart:
        # Native walker/RNG reload qualifies reconstruction, not exact DMC branch-state replay.
        continuation = output / "modern_probe.s001.cont.xml"
        if not continuation.is_file() or ET.parse(continuation).getroot().find("mcwalkerset") is None:
            raise RuntimeError("Missing native walker continuation checkpoint")
        with (output / "restart.log").open("w") as log:
            result = subprocess.run(command[:-1] + [continuation.name], cwd=output, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=600)
        if result.returncode:
            raise RuntimeError(f"Restart exit {result.returncode}; inspect {output / 'restart.log'}")
        restart_log = (output / "restart.log").read_text()
        if "Restart from the random number streams from the previous configuration." not in restart_log:
            raise RuntimeError("Native continuation did not restore the checkpoint RNG streams")
        if not re.search(r"HDFWalkerInput[^\n]*getting\s+[1-9]\d*\s+walkers", restart_log):
            raise RuntimeError("Native continuation did not load a positive walker population")
    sections = 4 if args.restart else 2
    for series in range(sections):
        scalar = output / f"modern_probe.s{series:03d}.scalar.dat"
        rows = [line.split() for line in scalar.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        if len(rows) != 2 or not all(math.isfinite(float(value)) for row in rows for value in row):
            raise RuntimeError(f"Expected two finite scalar blocks: {scalar}")
    if args.mode == "estimators":
        for series in range(sections):
            verify_estimators(output / f"modern_probe.s{series:03d}.stat.h5")
    print(f"PASS {args.mode}: {sections} modern sections, finite observations, resumed sampling; {output}")


if __name__ == "__main__":
    main()
