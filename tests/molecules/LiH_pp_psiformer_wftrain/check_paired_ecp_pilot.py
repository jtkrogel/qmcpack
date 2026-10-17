#!/usr/bin/env python3
"""Bounded prepared periodic LiH BFD paired ENERGY Adam pilot; no convergence claim."""

import argparse, csv, hashlib, json, math, os, re, signal, subprocess, time
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for name in ("binary", "input", "output"):
    p.add_argument("--" + name, type=Path, required=True)
p.add_argument("--source-commit", required=True)
p.add_argument(
    "--import-model",
    type=Path,
    help="fresh version0 training values, not a sampler/Hamiltonian restart",
)
a = p.parse_args()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(value):
    value = np.asarray(value).reshape(-1)[0]
    return value.decode() if isinstance(value, bytes) else str(value)


a.output.mkdir(parents=True, exist_ok=False)
tree = ET.parse(a.input)
root = tree.getroot()
head = root.find("qmcsystem/wavefunction/psiformer")
assert head.get("system") == "pseudopotential" and head.get("gpu") == "cuda"
assert (
    head.get("initialization") == "psiformer_complex_periodic_v1"
    and head.get("initialization_seed") == "37"
)
assert head.get("twist") == "0 0 0" and root.find("random").get("seed") == "91427"
cell = root.find("qmcsystem/simulationcell")
assert cell.find("parameter[@name='LR_handler']").text == "ewald"
assert cell.find("parameter[@name='LR_dim_cutoff']").text == "30"
training = root.find("qmc/training")
assert training.get("objective") == "energy" and training.get("optimizer") == "adam"
assert training.find("parameter[@name='learning_rate']").text.strip() == "1e-10"
protected = {str(a.input.resolve()): sha(a.input)}
imported_values = None
if a.import_model:
    model = a.import_model.resolve(strict=True)
    protected[str(model)] = sha(model)
    with h5py.File(model) as archive:
        assert text(archive["psiformer_model/system"][()]) == "pseudopotential"
        assert (
            int(
                np.asarray(archive["psiformer_model/paired_complex"][()]).reshape(-1)[0]
            )
            == 1
        )
        imported_values = archive["values"][()]
        assert np.isfinite(imported_values).all()
    # Keep current PP/XML/MC controls: standalone values are fresh initialization only.
    for key in ("initialization", "initialization_seed"):
        del head.attrib[key]
    head.set("parameters", str(model))
for pseudo in root.findall("qmcsystem/hamiltonian/pairpot/pseudo"):
    path = (a.input.parent / pseudo.get("href")).resolve()
    protected[str(path)] = sha(path)
    pseudo.set("href", str(path))
assert len(protected) == (4 if a.import_model else 3)


def parameter(name, value):
    node = training.find(f"parameter[@name='{name}']")
    if node is None:
        node = ET.SubElement(training, "parameter", name=name)
    node.text = str(value)


parameter("iterations", 2)
parameter(
    "checkpoint_interval", 1
)  # Artifact observability only, no scientific-control change.
tree.write(a.output / "input.xml", encoding="utf-8", xml_declaration=True)
report = dict(
    status="FAIL",
    source=a.source_commit,
    binary_sha256=sha(a.binary),
    protected=protected,
    helper_sha256=sha(Path(__file__)),
    scope="pilot_only_no_restart_or_convergence_claim",
)
env = {
    key: value
    for key, value in os.environ.items()
    if not key.startswith("PSIFORMER_GPU_")
}
env.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OMPI_MCA_memory="^patcher",
    OMPI_MCA_pml="ob1",
    OMPI_MCA_btl="self,tcp",
    PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES=str(2 * 1024**3),
    PSIFORMER_GPU_EXTERNAL_RESERVE_BYTES=str(256 * 1024**2),
    PSIFORMER_GPU_STATUS_CHECKS="synchronous",
)
child = None


def stop(signum, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    raise KeyboardInterrupt(f"signal {signum}")


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
started = time.monotonic()
try:
    with (a.output / "console.log").open("w") as stream:
        child = subprocess.Popen(
            [
                "/usr/bin/time",
                "-v",
                "-o",
                "resource_time.txt",
                str(a.binary.resolve()),
                "input.xml",
            ],
            cwd=a.output,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise AssertionError("paired ECP native pilot exceeded60s")
    report.update(returncode=code, wall_seconds=time.monotonic() - started)
    log = (a.output / "console.log").read_text()
    assert code == 0 and "NOT_RUN" not in log, log[-6000:]
    receipts = re.findall(
        r"WFTrain paired accepted-grid ECP: Q=(\d+) grid=(\d+) version=(\d+) samples=(\d+) max_abs_ImVnl=([^\s]+)",
        log,
    )
    assert len(receipts) == 2 and [int(r[2]) for r in receipts] == [0, 1]
    assert all(
        int(q) > 0
        and int(grid) > 0
        and int(samples) == 1
        and math.isfinite(float(im))
        and float(im) >= 0
        for q, grid, version, samples, im in receipts
    )
    with next(a.output.glob("*.training_info.csv")).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and [int(r["parameter_version"]) for r in rows] == [1, 2]
    assert all(
        r["optimizer"] == "adam"
        and r["objective"] == "energy"
        and int(r["committed"]) == 1
        and int(r["samples"]) == 1
        and all(
            math.isfinite(float(r[k]))
            for k in ("mean_energy", "variance", "gradient_norm")
        )
        for r in rows
    )
    models = sorted(a.output.rglob("*.parameters.h5"))
    assert len(models) == 2
    values = []
    for model in models:
        with h5py.File(model) as h:
            assert text(h["psiformer_model/system"][()]) == "pseudopotential"
            assert (
                int(np.asarray(h["psiformer_model/paired_complex"][()]).reshape(-1)[0])
                == 1
            )
            v = h["values"][()]
            assert np.isfinite(v).all()
            values.append(v)
    delta = values[1] - values[0]
    assert np.any(delta != 0), "No trainable bit changed between accepted versions1and2"
    delta_norm = float(np.linalg.norm(delta))
    assert np.isfinite(delta).all() and math.isfinite(delta_norm)
    if imported_values is not None:
        assert values[0].shape == imported_values.shape
        changed_from_import = int(np.count_nonzero(values[0] != imported_values))
        assert changed_from_import > 0, (
            "No trainable value changed on the first fresh imported-model update"
        )
        report.update(
            imported_model_sha256=protected[str(model)],
            fresh_initialization=True,
            first_update_changed_values=changed_from_import,
            scope="fresh_version0_current_Hamiltonian_not_sampler_or_PP_provenance_restart",
        )
    assert all(sha(Path(path)) == digest for path, digest in protected.items())
    rss = re.search(
        r"Maximum resident set size \(kbytes\): (\d+)",
        (a.output / "resource_time.txt").read_text(),
    )
    report.update(
        status="PASS_PILOT",
        maximum_rss_kib=int(rss.group(1)),
        receipts=receipts,
        changed_parameter_values=int(np.count_nonzero(delta)),
        delta_norm=delta_norm,
        actual_ImVnl=[float(r[-1]) for r in receipts],
        exact_restart_qualified=False,
    )
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
