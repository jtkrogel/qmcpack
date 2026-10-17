#!/usr/bin/env python3
"""Finite, opt-in Task45 CPU profile controller; default operation is inert.

Use an explicitly approved manifest and an existing Slurm allocation. Every
invocation executes one case only. Reports distinguish charged cgroup usage,
simultaneous process RSS and individual process high-water RSS. This is not a
performance CTest, scheduler or autotuner.
"""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def terminate(process):
    """Stop only the new session created by this controller/rank wrapper."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def cgroup_snapshot(group):
    counters = {key: (group / key).read_text().strip()
                for key in ("memory.current", "memory.peak", "memory.max", "memory.events")}
    processes = {}
    # This is the simultaneous RSS of ALL processes charged to this step-user
    # domain, including wrappers. Shared mappings can be counted more than once.
    for entry in group.rglob("cgroup.procs"):
        for number in entry.read_text().split():
            try:
                proc = Path("/proc") / number
                processes[number] = {"comm": (proc / "comm").read_text().strip(),
                                     "rss": int((proc / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")}
            except (FileNotFoundError, ProcessLookupError):
                pass
    return dict(time=time.time(), processes=processes,
                simultaneous_rss_sum=sum(value["rss"] for value in processes.values()), **counters)


def rank_run(plan_path):
    """Fail closed on the actual shared Slurm memory fence before child launch."""
    plan = json.loads(plan_path.read_text())
    rank = int(os.environ["SLURM_PROCID"])
    relative = Path("/proc/self/cgroup").read_text().strip().split(":", 2)[2]
    group = (Path("/sys/fs/cgroup") / relative.lstrip("/")).parent
    assert group.name == "user", (group, "unrecognized Slurm cgroup hierarchy")
    assert (group / "memory.max").read_text().strip() == str(plan["caps"]["aggregate_cgroup_bytes"])
    save(Path(f"rank_{rank}.launch.json"), dict(cgroup=str(group), rank=rank,
         affinity=sorted(os.sched_getaffinity(0)), step=os.environ["SLURM_STEP_ID"], plan=plan))
    command = ["/usr/bin/time", "-v", "-o", f"rank_{rank}.resource_time.txt"] + plan["command"]
    process = subprocess.Popen(command, start_new_session=True)
    def interrupted(number, frame):
        raise SystemExit(128 + number)
    signal.signal(signal.SIGTERM, interrupted)
    started, stop_reason = time.monotonic(), None
    try:
        with Path(f"rank_{rank}.memory.jsonl").open("w", buffering=1) as trace:
            while True:
                snapshot = cgroup_snapshot(group)
                trace.write(json.dumps(snapshot) + "\n")
                if max(int(snapshot["memory.current"]), int(snapshot["memory.peak"])) >= plan["caps"]["stop_observed_bytes"]:
                    stop_reason = "80 percent charged-memory threshold"
                if time.monotonic() - started >= plan["caps"]["case_seconds"]:
                    stop_reason = "90 second case deadline"
                if stop_reason:
                    terminate(process)
                if process.poll() is not None:
                    break
                time.sleep(0.25)
    finally:
        terminate(process)
    save(Path(f"rank_{rank}.finish.json"), dict(exit_code=process.returncode,
         elapsed=time.monotonic() - started, stop_reason=stop_reason, final=cgroup_snapshot(group)))
    return process.returncode if not stop_reason else 124


def validate_manifest(manifest):
    caps = manifest["caps"]
    assert 0 < caps["case_seconds"] <= 90 and 0 < caps["campaign_seconds"] <= 900
    assert 0 < caps["maximum_attempts"] <= 10 and caps["automatic_retry"] is False
    assert caps["aggregate_cgroup_bytes"] == 14 * 1024**3
    assert 0 < caps["stop_observed_bytes"] <= int(0.8 * caps["aggregate_cgroup_bytes"])
    identifiers = [case["id"] for case in manifest["cases"]]
    assert len(identifiers) == len(set(identifiers)) <= 10
    assert all(identifier and all(c.isalnum() or c in "_-" for c in identifier) for identifier in identifiers)
    for case in manifest["cases"]:
        assert case["request"] in ("value", "complete_vgl", "native_actual_ecp_adam", "native_vmc")
        assert 1 <= case["walkers"] <= 4
        assert case.get("ranks", 1) == (2 if case["request"] == "native_actual_ecp_adam" else 1)
        assert case.get("configuration_tile", 1) in (1, 2)
        assert case.get("target_tile", "FULL") in ("FULL", 1)
        if case["request"] == "native_vmc":
            assert case["walkers"] == 1, "native K1 pilot is W1 only"


def validate_native_vmc(directory):
    """Validate this one-block physical W1 pilot, not an optimizer CSV."""
    log = (directory / "console.log").read_text()
    assert "CPU target VGL: maxK=1," in log and "CPU target VGL crowd: live_clones=1," in log
    scalar, = directory.glob("*.scalar.dat")
    lines = scalar.read_text().splitlines()
    header = next(line.split()[1:] for line in lines if line.startswith("#") and "LocalEnergy" in line)
    data = [line.split() for line in lines if line.strip() and not line.startswith("#")]
    assert len(data) == 1 and len(data[0]) == len(header), "require one complete native scalar block"
    values = dict(zip(header, map(float, data[0])))
    assert all(math.isfinite(value) for value in values.values()), "nonfinite native VMC scalar"
    assert values["BlockWeight"] == 1 and 0 < values["AcceptRatio"] <= 1, "native population/moves mismatch"
    assert math.isfinite(values["LocalEnergy_sq"] - values["LocalEnergy"]**2)
    assert not list(directory.glob("*.training*")), "VMC pilot must not train"
    return values


def compare_values(first, second):
    """Compare ALL complete-request results, not a checksum or first electron."""
    left, right = (json.loads(Path(path).read_text()) for path in (first, second))
    for key in ("configurations_completed", "electrons_per_configuration"):
        assert left[key] == right[key], key
    assert len(left["values"]) == len(right["values"]) > 0
    maximum = 0.0
    for a, b in zip(left["values"], right["values"]):
        assert math.isfinite(a) and math.isfinite(b)
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10), (a, b)
        maximum = max(maximum, abs(a - b))
    return dict(compared_values=len(left["values"]), maximum_absolute_difference=maximum,
                relative_tolerance=1e-9, absolute_tolerance=1e-10)


def execute(manifest, manifest_path, identifier):
    validate_manifest(manifest)
    assert manifest["status"] == "ready_to_execute", "Root has not authorized this immutable manifest"
    assert identifier in manifest["approved_case_ids"], "Root has not approved this case launch"
    root = Path(manifest["runtime_root"]).resolve()
    assert manifest_path.parent.resolve() == root
    allocation = str(manifest["allocation_id"])
    assert allocation.isdigit() and manifest["allocation_verified_utc"] and manifest["node"]
    case = next(case for case in manifest["cases"] if case["id"] == identifier)
    native = case["request"] in ("native_actual_ecp_adam", "native_vmc")
    executable = Path(manifest["application_executable"] if native else manifest["benchmark_executable"])
    source = manifest["application_source" if native else "benchmark_source"]
    assert len(source) == 40 and all(c in "0123456789abcdef" for c in source)
    assert sha256(executable) == manifest["application_sha256" if native else "benchmark_sha256"]
    model = manifest["model_inputs"][case["model"]]
    for key in ("parameter", "configuration"):
        assert sha256(model[key + "_path"]) == model[key + "_sha256"]
    for path, expected in manifest.get("auxiliary_input_hashes", {}).items():
        assert sha256(path) == expected, path
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger_path = root / "status.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"started": time.time(), "attempts": []}
    assert identifier not in ledger["attempts"], "No automatic or repeated case execution"
    assert len(ledger["attempts"]) < manifest["caps"]["maximum_attempts"]
    remaining = manifest["caps"]["campaign_seconds"] - (time.time() - ledger["started"])
    assert remaining >= manifest["caps"]["case_seconds"], "Insufficient aggregate budget for another case"
    ledger["attempts"].append(identifier)
    save(ledger_path, ledger)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "cases" / identifier / "attempts" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    if native:
        original = Path(manifest["original_native_xml"])
        assert sha256(original) == case["original_xml_sha256"]
        # Preserve the original four-level relative asset layout via root/inputs.
        shutil.copyfile(original, directory / "qmc.xml")
        command = [str(executable), "qmc.xml"]
    else:
        command = [str(executable), "--scenario", "batch_value" if case["request"] == "value" else "complete_vgl",
                   "--parameters", model["parameter_path"], "--configuration", model["configuration_path"],
                   "--walkers", str(case["walkers"]), "--warm-calls", "0", "--output", "report.json",
                   "--configuration-tile", str(case.get("configuration_tile", 1)),
                   "--target-tile", str(0 if case.get("target_tile", "FULL") == "FULL" else case["target_tile"]),
                   "--open-coulomb", "1" if model.get("generated") else "0",
                   "--owner-budget", str(10 * 1024**3)]
    plan = dict(command=command, caps=manifest["caps"], case=case, source=source)
    environment = dict(os.environ, OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1")
    if case["request"] == "native_vmc":
        controls = dict(PSIFORMER_CPU_VGL_TARGETS="1", PSIFORMER_CPU_TARGET_VGL_OWNER_BYTES_PER_CLONE="16777216",
                        **{f"PSIFORMER_{mode}_BACKEND": "direct" for mode in ("VALUE", "SPATIAL", "SCORE", "KINETIC")})
        environment.update(controls)
        for key in ("PSIFORMER_CPU_KINETIC_TARGETS", "PSIFORMER_CPU_TARGET_KINETIC_OWNER_BYTES_PER_LANE"):
            environment.pop(key, None)
        plan["native_environment"] = controls
    save(directory / "rank_plan.json", plan)
    if manifest.get("runtime_library_path"):
        environment["LD_LIBRARY_PATH"] = manifest["runtime_library_path"] + ":" + environment.get("LD_LIBRARY_PATH", "")
    launch = ["srun", "--jobid=" + allocation, "--exclusive", "--exact", "--nodes=1",
              "--nodelist=" + manifest["node"], "--ntasks=" + str(case.get("ranks", 1)),
              "--cpus-per-task=2", "--mem=14G", "--cpu-bind=cores", "--distribution=block:block",
              "--kill-on-bad-exit=1", "--time=00:02:00", "--export=ALL", sys.executable,
              str(Path(__file__).resolve()), "--rank", str(directory / "rank_plan.json")]
    save(directory / "launch.json", dict(manifest=manifest, manifest_sha256=sha256(manifest_path),
         controller_sha256=sha256(__file__), command=launch))
    started = time.monotonic()
    with (directory / "console.log").open("w") as output:
        try:
            process = subprocess.Popen(launch, cwd=directory, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            save(directory / "result.json", dict(exit_code=127, elapsed=time.monotonic() - started,
                 directory=str(directory), launch_error=str(error)))
            return 127
        try:
            code = process.wait(timeout=manifest["caps"]["case_seconds"])
        except subprocess.TimeoutExpired:
            terminate(process)
            code = 124
        finally:
            terminate(process)
    result = dict(exit_code=code, elapsed=time.monotonic() - started, directory=str(directory))
    try:
        if not code and not native:
            values = json.loads((directory / "report.json.values.json").read_text())
            assert values["configurations_completed"] == case["walkers"], "incomplete configuration count"
            assert all(math.isfinite(number) for number in values["values"]), "nonfinite output"
            lanes = (2 if case["request"] == "value" else
                     3 + 5 * values["electrons_per_configuration"] + int(bool(model.get("generated"))))
            assert len(values["values"]) == case["walkers"] * lanes, "incomplete output lanes"
        if not code and case["request"] == "native_vmc":
            result["native_scalars"] = validate_native_vmc(directory)
            for key in ("parameter", "configuration"):
                assert sha256(model[key + "_path"]) == model[key + "_sha256"], "protected native input changed"
        elif not code and native:
            csv_path, = directory.glob("*.training_info.csv")
            rows = list(csv.DictReader(csv_path.open()))
            assert len(rows) == 1 and rows[0]["committed"] == "1" and rows[0]["samples"] == "4", "native work mismatch"
            assert all(math.isfinite(float(rows[0][key])) for key in ("mean_energy", "variance", "gradient_norm")), "nonfinite native result"
    except (OSError, ValueError, KeyError, AssertionError) as error:
        result["validation_error"] = str(error)
        result["exit_code"] = code = 1
    save(directory / "result.json", result)
    print(json.dumps(result), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--execute-case")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("FIRST", "SECOND"))
    parser.add_argument("--rank", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rank:
        return rank_run(args.rank)
    if args.compare:
        print(json.dumps(compare_values(*args.compare)))
        return 0
    if not args.manifest:
        parser.error("manifest is required")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if not args.execute_case:
        print(json.dumps(dict(status=manifest["status"], execution="not requested", cases=len(manifest["cases"]))))
        return 0
    return execute(manifest, args.manifest.resolve(), args.execute_case)


if __name__ == "__main__":
    sys.exit(main())
