#!/usr/bin/env python3
"""Reuse the protected paired ECP pilot for ONE exact initialized split/resume."""

import argparse, csv, json, math, os, re, runpy, signal, subprocess, time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for name in ("binary", "source", "pilot-report", "output"):
    p.add_argument("--" + name, type=Path, required=True)
p.add_argument(
    "--pilot-helper",
    type=Path,
    help="registered pilot helper; manual default remains beside the evidence",
)
a = p.parse_args()
utility = runpy.run_path(
    str(
        a.source
        / "tests/molecules/LiH_ae_psiformer_gpu/check_resident_device_authority.py"
    )
)
sha, fields = (utility[k] for k in ("sha", "fields"))
envelopes = runpy.run_path(
    str(a.source / "tests/molecules/LiH_ae_psiformer_gpu/check_rejected_full_reuse.py")
)["envelopes"]
pilot = json.loads(a.pilot_report.read_text())
continuous = a.pilot_report.parent
assert pilot["status"] == "PASS_PILOT" and sha(a.binary) == pilot["binary_sha256"]
assert (
    subprocess.check_output(
        ["git", "-C", str(a.source), "rev-parse", "HEAD"], text=True
    ).strip()
    == pilot["source"]
)
assert all(sha(Path(path)) == value for path, value in pilot["protected"].items())
protected = {str(path): sha(path) for path in continuous.rglob("*") if path.is_file()}
assert (
    sha(a.pilot_helper or continuous.parent / "check_paired_ecp_native_A.py")
    == pilot["helper_sha256"]
)
a.output.mkdir(parents=True, exist_ok=False)
report = dict(
    status="FAIL",
    source=pilot["source"],
    binary_sha256=sha(a.binary),
    helper_sha256=sha(Path(__file__)),
    pilot_report_sha256=sha(a.pilot_report),
    cases={},
)
deadline = time.monotonic() + 180
child = None


def stop(signum, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    raise KeyboardInterrupt(f"signal {signum}")


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
env = {k: v for k, v in os.environ.items() if not k.startswith("PSIFORMER_GPU_")}
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


def history(folder):
    with next(folder.glob("*.training_info.csv")).open() as stream:
        return list(csv.DictReader(stream))


def run(name, iterations, restart=None):
    global child
    folder = a.output / name
    folder.mkdir()
    tree = ET.parse(continuous / "input.xml")
    training = tree.find("qmc/training")
    training.find("parameter[@name='iterations']").text = str(iterations)
    if restart:
        ET.SubElement(training, "parameter", name="restart_path").text = str(
            restart.resolve()
        )
    tree.write(folder / "input.xml", encoding="utf-8", xml_declaration=True)
    started = time.monotonic()
    with (folder / "console.log").open("w") as stream:
        child = subprocess.Popen(
            [
                "/usr/bin/time",
                "-v",
                "-o",
                "resource_time.txt",
                str(a.binary.resolve()),
                "input.xml",
            ],
            cwd=folder,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=max(0.001, min(60, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise AssertionError("split/resume exceeded60s child/180s aggregate")
    log = (folder / "console.log").read_text()
    report["cases"][name] = dict(
        returncode=code,
        wall_seconds=time.monotonic() - started,
        input_sha256=sha(folder / "input.xml"),
    )
    assert code == 0 and "NOT_RUN" not in log, log[-6000:]
    rows = history(folder)
    assert len(rows) == 1
    row = rows[0]
    assert (
        row["objective"],
        row["optimizer"],
        int(row["committed"]),
        int(row["samples"]),
    ) == ("energy", "adam", 1, 1)
    assert (
        int(row["parameter_version"]) == iterations
        and int(row["sample_version"]) == iterations - 1
    )
    assert all(
        math.isfinite(float(row[k]))
        for k in ("mean_energy", "variance", "gradient_norm")
    )
    receipts = re.findall(
        r"WFTrain paired accepted-grid ECP: Q=(\d+) grid=(\d+) version=(\d+) samples=(\d+) max_abs_ImVnl=([^\s]+)",
        log,
    )
    assert len(receipts) == 1
    q, grid, version, samples, im = receipts[0]
    assert (
        int(q) > 0
        and int(grid) > 0
        and int(version) == iterations - 1
        and int(samples) == 1
        and math.isfinite(float(im))
    )
    report["cases"][name].update(
        receipt=receipts[0],
        maximum_rss_kib=int(
            re.search(
                r"Maximum resident set size \(kbytes\): (\d+)",
                (folder / "resource_time.txt").read_text(),
            ).group(1)
        ),
    )
    return folder, rows


def exact(left, right):
    l, r = fields(left), fields(right)
    assert l.keys() == r.keys()
    for key in l:
        np.testing.assert_array_equal(l[key], r[key], err_msg=key)


def checkpoint(folder):
    return envelopes(next(folder.glob("*.last_checkpoint.h5")))


try:
    split, first = run("split_one", 1)
    protected_split = {
        str(path): sha(path) for path in split.rglob("*") if path.is_file()
    }
    resumed, last = run("resumed_two", 2, next(split.glob("*.last_checkpoint.h5")))
    for left, right in zip(checkpoint(continuous), checkpoint(resumed)):
        exact(left, right)
    for reference, actual in (
        (
            sorted(continuous.rglob("*.parameters.h5"))[0],
            next(split.rglob("*.parameters.h5")),
        ),
        (
            sorted(continuous.rglob("*.parameters.h5"))[-1],
            next(resumed.rglob("*.parameters.h5")),
        ),
    ):
        exact(reference, actual)
    reference_rows = history(continuous)
    assert len(reference_rows) == len(first + last) == 2
    for left, right in zip(reference_rows, first + last):
        assert left.keys() == right.keys()
        assert all(
            left[k] == right[k]
            for k in left
            if k not in {"sampling_seconds", "total_seconds", "resident_bytes"}
        )
    assert all(sha(Path(path)) == value for path, value in protected.items())
    assert all(sha(Path(path)) == value for path, value in protected_split.items())
    assert all(sha(Path(path)) == value for path, value in pilot["protected"].items())
    assert time.monotonic() < deadline
    report.update(
        status="PASS_EXACT_INITIALIZED_RESTART",
        protected_pilot_files=protected,
        protected_split_files=protected_split,
        scope="Exact ROOT/sampler/all parameter datasets/nonclock history; no convergence claim",
    )
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
