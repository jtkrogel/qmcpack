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
p.add_argument("--reference", action="store_true", help="W2 same-grid reference selection; natural rejection is valid")
p.add_argument(
    "--import-model",
    type=Path,
    help="fresh version0 training values, not a sampler/Hamiltonian restart",
)
a = p.parse_args()
if a.reference and a.import_model:
    p.error("--reference preserves the internally initialized qualified model")


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
    imported_model_path = a.import_model.resolve(strict=True)
    protected[str(imported_model_path)] = sha(imported_model_path)
    with h5py.File(imported_model_path) as archive:
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
    head.set("parameters", str(imported_model_path))
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
if a.reference:
    root.find("qmc/parameter[@name='walkers_per_rank']").text = "2"
    for name, value in (("line_search", "reference"), ("line_search_maximum_weight", ".75"),
                        ("line_search_minimum_ess_fraction", ".5"), ("line_search_confidence", "2")):
        parameter(name, value)
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
    assert len(receipts) == 2
    assert all(
        int(q) > 0
        and int(grid) > 0
        and int(samples) == (2 if a.reference else 1)
        and math.isfinite(float(im))
        and float(im) >= 0
        for q, grid, version, samples, im in receipts
    )
    with next(a.output.glob("*.training_info.csv")).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    commits = [int(row["committed"]) for row in rows]
    assert all(value in (0, 1) for value in commits)
    versions = np.cumsum(commits).tolist()
    assert [int(row["parameter_version"]) for row in rows] == versions
    assert [int(row["sample_version"]) for row in rows] == [0, versions[0]]
    assert [int(receipt[2]) for receipt in receipts] == [0, versions[0]]
    if not a.reference:
        assert commits == [1, 1]
    assert all(
        r["optimizer"] == "adam"
        and r["objective"] == "energy"
        and int(r["samples"]) == (2 if a.reference else 1)
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
    if commits[1]:
        assert np.any(delta != 0), "No trainable bit changed on the second accepted update"
    else:
        np.testing.assert_array_equal(values[1], values[0])
    delta_norm = float(np.linalg.norm(delta))
    assert np.isfinite(delta).all() and math.isfinite(delta_norm)
    if a.reference:
        assessments = [tuple(map(float, match)) for match in re.findall(
            r"WFTrain candidate alpha (\S+) loss (\S+) loss_difference (\S+) difference_error (\S+) "
            r"ess (\S+) maximum_weight (\S+) valid ([01])", log)]
        assert len(assessments) == 12
        complete = re.findall(r"WFTrain paired fixed-grid candidate: alpha=(\S+) sample=(\d+) grid=(\d+) "
            r"source_version=(\d+) candidate_version=(\d+) ReEL=(\S+) ImEL=(\S+) no_rng_draw=yes", log)
        assert len(complete) == 28  # Baseline refresh plus six assessed scales, two samples/iteration.
        states = sorted(a.output.rglob("*.state.h5"))
        assert len(states) == 2
        previous_state = None
        for iteration, row in enumerate(rows):
            candidates = assessments[6 * iteration:6 * (iteration + 1)]
            assert [entry[0] for entry in candidates] == [0, 1, .125, .25, .5, .75]
            assert all(math.isfinite(value) for entry in candidates for value in entry)
            assert all(0 < entry[4] <= 2 and 0 < entry[5] <= 1 for entry in candidates)
            alpha = float(row["selected_alpha"])
            eligible = [entry for entry in candidates if entry[0] == alpha]
            assert eligible and int(alpha != 0) == commits[iteration]
            if alpha:
                selected = eligible[0]
                assert selected[6] == 1 and selected[1] - candidates[0][1] + 2 * selected[3] < 0
            else:
                assert not any(entry[0] and entry[6] and entry[1] - candidates[0][1] + 2 * entry[3] < 0
                               for entry in candidates)
            observed = complete[14 * iteration:14 * (iteration + 1)]
            assert [float(entry[0]) for entry in observed[::2]] == [0, 0, 1, .125, .25, .5, .75]
            assert [int(entry[1]) for entry in observed] == [0, 1] * 7
            assert all(int(entry[2]) == int(receipts[iteration][1]) and int(entry[3]) == int(receipts[iteration][2])
                       and math.isfinite(float(entry[5])) and math.isfinite(float(entry[6])) for entry in observed)
            assert [int(entry[4]) for entry in observed] == [int(receipts[iteration][2]) + 1 + i // 2 for i in range(14)]
            initial = np.asarray([[float(entry[5]), float(entry[6])] for entry in observed[:2]])
            repeated = np.asarray([[float(entry[5]), float(entry[6])] for entry in observed[2:4]])
            # Same source aliases/grid must reproduce the native training objective, not merely finite trials.
            np.testing.assert_allclose(repeated, initial, rtol=3e-9, atol=3e-9)
            np.testing.assert_allclose([candidates[0][1], float(row["mean_energy"])], initial[:, 0].mean(),
                                       rtol=3e-9, atol=3e-9)
            with h5py.File(states[iteration]) as archive:
                group = archive["wftrain_checkpoint"]
                np.testing.assert_array_equal(group["model/values"][()], values[iteration])
                count = int(np.asarray(group["optimizer/accepted_update_count"][()]).reshape(-1)[0])
                assert count == versions[iteration]
                state = {name: group[f"optimizer/{name}"][()] for name in ("first_moment", "second_moment")}
                assert all(np.isfinite(value).all() for value in state.values())
                if not commits[iteration]:
                    for name, value in state.items():
                        np.testing.assert_array_equal(value, previous_state[name] if previous_state else np.zeros_like(value))
                previous_state = state
        report.update(selected_alphas=[float(row["selected_alpha"]) for row in rows],
                      committed_updates=sum(commits), complete_candidates=complete,
                      reference=True, accepted_parameter_versions=versions,
                      scope="W2_same_grid_REFERENCE_natural_decisions_no_convergence_claim",
                      initial_parameter_preservation="first-zero-step uses existing selector proof; adjacent native zero-step exact")
    if imported_values is not None:
        assert values[0].shape == imported_values.shape
        changed_from_import = int(np.count_nonzero(values[0] != imported_values))
        assert changed_from_import > 0, (
            "No trainable value changed on the first fresh imported-model update"
        )
        report.update(
            imported_model_sha256=protected[str(imported_model_path)],
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
