#!/usr/bin/env python3
"""Bounded native SJ update/inference regression; retain all evidence in a temp dir."""
import argparse
import csv
import math
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def datasets(path):
    result = {}
    with h5py.File(path, "r") as handle:
        handle.visititems(lambda name, obj: result.update({name: obj[()]})
                          if isinstance(obj, h5py.Dataset) else None)
    return result


def compare_archives(left, right, pattern):
    first = sorted(left.rglob(pattern))
    second = sorted(right.rglob(pattern))
    assert first and len(first) == len(second), (pattern, first, second)
    for a, b in zip(first, second):
        lhs, rhs = datasets(a), datasets(b)
        assert lhs.keys() == rhs.keys(), (a, b)
        for name in lhs:
            np.testing.assert_array_equal(lhs[name], rhs[name], err_msg=f"{pattern}: {name}")


def history(directory):
    with (directory / "sj.s000.training_info.csv").open() as handle:
        return list(csv.DictReader(handle))


def text_value(value):
    value = np.asarray(value).reshape(-1)[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def check_orbital_frame(group, tree, parameters, row, model):
    """Check the explicit two-generator model, not arbitrary registered vars."""
    wrapper = tree.find(".//rotated_sposet")
    assert wrapper is not None and wrapper.find("sposet").get("size") == "3"
    wrapper_name = wrapper.get("name")
    expected = [f"{wrapper_name}_orb_rot_0000_{virtual:04d}_r" for virtual in (1, 2)]
    names = [text_value(value) for value in np.asarray(group["variable_names"][()]).reshape(-1)]
    indices = np.asarray(group["active_indices"][()]).reshape(-1)
    assert sorted(name for name in names if "_orb_rot_" in name) == expected, (model, names)
    active = parameters[[int(indices[names.index(name)]) for name in expected]]
    frame = group["orbital_frame"]
    np.testing.assert_array_equal(frame["format_version"][()], [1, 0, 0])
    np.testing.assert_array_equal(frame["determinant_aliases"][()], [0, 0])
    assert int(np.asarray(frame["source_version"][()]).reshape(-1)[0]) == int(row["parameter_version"])
    assert int(np.asarray(frame["orbital_count"][()]).reshape(-1)[0]) == 1
    orbital = frame["spo_0"]
    assert text_value(orbital["name"][()]) == wrapper_name
    np.testing.assert_array_equal(orbital["shape"][()], [3, 3])
    initial = np.asarray([float(value) for value in wrapper.find("sposet/coefficient").text.split()])
    reference = np.asarray(orbital["reference_coefficients"][()]).reshape(-1)
    accepted = np.asarray(orbital["accepted_coefficients"][()]).reshape(-1)
    full = np.asarray(orbital["full_rotation"][()]).reshape(-1)
    np.testing.assert_array_equal(reference, initial)
    assert accepted.size == 9 and full.size == 3
    assert np.isfinite(accepted).all() and np.isfinite(full).all()
    if row["committed"] == "1":
        assert not np.array_equal(accepted, reference), model
    # Both bundle representations must carry exactly the same complete frame.
    exported = datasets(model)
    iteration_root = model.name[:-len(".parameters.h5")]
    state = datasets(model.parent / (iteration_root + ".checkpoint") / (iteration_root + ".state.h5"))
    prefix = "conventional_wavefunction/orbital_frame/"
    for name, values in exported.items():
        if name.startswith(prefix):
            np.testing.assert_array_equal(values, state["wftrain_checkpoint/orbital_frame/" + name[len(prefix):]],
                                          err_msg=f"orbital checkpoint/export frame: {name}")
    return active, full


def apply_model_reference(tree, model):
    """Construct a test-only explicit physical model, without the import path."""
    with h5py.File(model, "r") as handle:
        group = handle["conventional_wavefunction"]
        names = [text_value(value) for value in np.asarray(group["variable_names"][()]).reshape(-1)]
        indices = np.asarray(group["active_indices"][()]).reshape(-1)
        parameters = np.asarray(group["parameters"][()]).reshape(-1)
        active = {name: parameters[int(index)] for name, index in zip(names, indices) if index >= 0}
        for coefficients in tree.findall("wavefunction/jastrow/correlation/coefficients"):
            values = coefficients.text.split()
            coefficients.text = " ".join(str(active.get(f"{coefficients.get('id')}_{i}", float(value)))
                                         for i, value in enumerate(values))
        for node in tree.findall(".//multideterminant/detlist/*"):
            if node.get("id") in active:
                node.set("coeff", str(active[node.get("id")]))
        wrapper = tree.find(".//rotated_sposet")
        if wrapper is not None:
            # Remove the coordinate wrapper entirely: the independent VMC oracle
            # starts from exact accepted C, not an exp(active) reconstruction.
            spo = wrapper.find("sposet")
            spo.set("name", wrapper.get("name"))
            spo.find("coefficient").text = " ".join(str(value) for value in np.asarray(
                group["orbital_frame/spo_0/accepted_coefficients"][()]).reshape(-1))
            collection = tree.find("wavefunction/sposet_collection")
            collection.remove(wrapper)
            collection.append(spo)


def vmc_physical_trace(directory):
    """Read deterministic scalar energy/weight traces, excluding elapsed CPU."""
    path = directory / "sj.s000.scalar.dat"
    header = next(line.split()[1:] for line in path.read_text().splitlines() if line.startswith("#"))
    data = np.loadtxt(path, ndmin=2)
    selected = [i for i, name in enumerate(header) if name not in ("BlockCPU", "index")]
    assert selected and "LocalEnergy" in header, path
    result = data[:, selected]
    assert np.isfinite(result).all(), path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("fixture", type=Path, nargs="?", default=Path(__file__).with_name("wftrain_sj.xml"))
    parser.add_argument("--restart", action="store_true", help="also require split-run exact continuation")
    parser.add_argument("--preflight", action="store_true", help="also check unsupported model/optimizer rejection")
    parser.add_argument("--model-import", action="store_true",
                        help="export/import ordinary VMC, fresh training, re-export and exact restart")
    parser.add_argument("--companion-optimizer", choices=("sgd", "sr"), default="sgd",
                        help="second representative optimizer in the main workflow")
    parser.add_argument("--ci-representation", choices=("dets", "csf"), default="dets",
                        help="optionally express a paired DETS fixture as one-determinant CSFs")
    parser.add_argument("--orbital-jastrow", action="store_true",
                        help="add the qualified four-coefficient J2 to the orbital-only fixture")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--profile-only", action="store_true", help="one bounded Adam/inference case for MPI/OpenMP")
    parser.add_argument("--optimizer-screen", action="store_true", help="six bounded nonlinear SJ optimizer cases")
    parser.add_argument("--family-screen", action="store_true", help="bounded LM/ALM/PII/minPII cases; report rejection honestly")
    parser.add_argument("--family-method", choices=("lm", "alm", "pii", "minpii"), action="append",
                        help="restrict a family screen to selected methods, e.g. one corrected fixture")
    args = parser.parse_args()
    if args.threads <= 0 or args.ranks <= 0:
        parser.error("--threads and --ranks must be positive")
    if (args.profile_only or args.optimizer_screen or args.family_screen) and (args.restart or args.preflight):
        parser.error("profile/screen options select their cases; omit --restart/--preflight")
    if sum((args.profile_only, args.optimizer_screen, args.family_screen)) > 1:
        parser.error("select one of --profile-only, --optimizer-screen, --family-screen")
    if args.family_method and not args.family_screen:
        parser.error("--family-method requires --family-screen")
    if args.model_import and any((args.profile_only, args.optimizer_screen, args.family_screen,
                                  args.restart, args.preflight)):
        parser.error("--model-import selects its own bounded workflow")
    executable = args.executable.resolve()
    fixture = args.fixture.resolve()
    work = Path(tempfile.mkdtemp(prefix="conventional_sj_verify_"))
    print(work, flush=True)

    def run(label, *, optimizer="adam", inference=True, iterations=2, restart=None, mutate=None, reject=False,
            samples_per_rank=4, require_committed=True, parameter_file=None, vmc=False, reference_file=None):
        directory = work / label
        directory.mkdir()
        tree = ET.parse(fixture)
        # Recipes run in fresh evidence directories, not beside their PP asset.
        # Keep the tracked source untouched; scientific identity uses its bytes.
        for pseudo in tree.findall("hamiltonian/pairpot/pseudo"):
            href = Path(pseudo.get("href"))
            if not href.is_absolute():
                pseudo.set("href", str((fixture.parent / href).resolve()))
        orbital_model = tree.find(".//rotated_sposet") is not None
        if args.orbital_jastrow:
            assert orbital_model and tree.find("wavefunction/jastrow") is None, fixture
            tree.find("wavefunction").append(ET.fromstring('''
                <jastrow name="J2" type="Two-Body" function="Bspline">
                  <correlation rcut="4" size="4" speciesA="u" speciesB="d">
                    <coefficients id="ud" type="Array" optimize="yes">0.12 -0.08 0.04 -0.01</coefficients>
                  </correlation>
                </jastrow>'''))
        if args.ci_representation == "csf":
            expansion = tree.find("wavefunction/determinantset/multideterminant/detlist")
            assert expansion is not None and expansion.get("type") == "DETS", fixture
            expansion.set("type", "CSF")
            for old, new in (("nc0", "nca"), ("nc1", "ncb"), ("ne0", "nea"), ("ne1", "neb")):
                expansion.set(new, expansion.attrib.pop(old))
            for ordinal, node in enumerate(list(expansion)):
                alpha, beta = node.get("occ0"), node.get("occ1")
                csf = ET.Element("csf", id=node.get("id"), coeff=node.get("coeff"),
                                 occ="".join(str(int(a) + int(b)) for a, b in zip(alpha, beta)),
                                 exctLvl="0" if ordinal == 0 else "2")
                ET.SubElement(csf, "det", coeff="1", alpha=alpha, beta=beta)
                expansion.remove(node)
                expansion.append(csf)
        tree.find("project").set("id", "sj")
        qmc = tree.find("qmc")
        training = qmc.find("training")
        training.set("optimizer", optimizer)
        training.find("parameter[@name='iterations']").text = str(iterations)
        if not inference:
            qmc.remove(qmc.find("inference"))
        if restart is not None:
            ET.SubElement(training, "parameter", name="restart_path").text = str(restart)
        if parameter_file is not None:
            tree.find("wavefunction").set("parameter_file", str(parameter_file.resolve()))
        if reference_file is not None:
            apply_model_reference(tree, reference_file)
        if vmc:
            qmc.set("method", "vmc")
            qmc.remove(training)
            if qmc.find("inference") is not None:
                qmc.remove(qmc.find("inference"))
            qmc.find("parameter[@name='blocks']").text = "2"
        if mutate:
            mutate(tree)
        # Only the explicit qualified rotation wrapper may unfreeze orbitals.
        if not reject:
            if orbital_model:
                assert not tree.findall(".//sposet[@optimize='yes']")
            else:
                assert all(node.get("optimize") == "no" for node in tree.findall(".//sposet"))
        tree.write(directory / "run.xml")
        environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1")
        command = [str(executable), "run.xml"]
        if args.ranks > 1:
            command = [args.mpiexec, "-np", str(args.ranks)] + command
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=directory, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180)
        text = (directory / "run.log").read_text()
        if reject:
            assert result.returncode != 0, (directory, text[-4000:])
            assert reject in text, (directory, reject, text[-4000:])
            assert not list(directory.glob("*.training/*/*.state.h5")), directory
            return directory
        assert result.returncode == 0, (directory, text[-6000:])
        if vmc:
            vmc_physical_trace(directory)
            assert not list(directory.glob("*.training")), directory
            return directory
        rows = history(directory)
        expected_rows = iterations if restart is None else 1
        assert len(rows) == expected_rows, (directory, rows)
        for row in rows:
            if require_committed:
                assert row["committed"] == "1", (directory, row)
            assert row["committed"] in ("0", "1"), (directory, row)
            assert int(row["samples"]) == samples_per_rank * args.ranks, (directory, row)
            assert all(math.isfinite(float(row[name])) for name in ("mean_energy", "variance", "gradient_norm")), row
            if require_committed:
                assert float(row["gradient_norm"]) > 0, row
            assert int(row["parameter_version"]) == int(row["sample_version"]) + int(row["committed"]), row
        assert (directory / "sj.s000.last_checkpoint.h5").is_file(), directory
        family = directory / "sj.s000.inference"
        files = sorted(family.glob("*.scalar.dat"))
        assert len(files) == (expected_rows if inference else 0), (directory, files)
        for path, row in zip(files, rows):
            scalar = path.read_text()
            iteration = int(row["iteration"])
            assert f"iteration {iteration} parameter_version {row['parameter_version']} " in scalar, path
            assert "complete yes" in scalar, path
            data = [line.split() for line in scalar.splitlines() if line.strip() and not line.startswith("#")]
            assert len(data) == 1, path
            header = next(line.split()[1:] for line in scalar.splitlines() if line.startswith("#") and "BlockWeight" in line)
            assert float(data[0][header.index("BlockWeight")]) == 2 * args.ranks, path
        models = sorted((directory / "sj.s000.training").rglob("*.parameters.h5"))
        assert len(models) == expected_rows, (directory, models)
        determinant_list = tree.find("wavefunction/determinantset/multideterminant/detlist")
        linear_nodes = [] if determinant_list is None else list(determinant_list)
        initial = [float(node.get("coeff")) for node in linear_nodes[1:]]
        if orbital_model:
            initial += [0.0, 0.0]
        initial += [float(value) for node in tree.findall("wavefunction/jastrow/correlation/coefficients")
                   for value in node.text.split()]
        previous = (np.asarray(initial) if parameter_file is None else
                    datasets(parameter_file)["conventional_wavefunction/parameters"].reshape(-1))
        fixed = tree.find(".//sposet/coefficient")
        orbital_history = []
        for model, row in zip(models, rows):
            with h5py.File(model, "r") as handle:
                group = handle["conventional_wavefunction"]
                parameters = np.asarray(group["parameters"][()]).reshape(-1)
                assert parameters.size == len(initial) and np.isfinite(parameters).all(), model
                if linear_nodes:
                    assert parameters.size == len(initial) and parameters.size > len(linear_nodes) - 1
                    names = [value.decode() if isinstance(value, bytes) else str(value)
                             for value in np.asarray(group["variable_names"][()]).reshape(-1)]
                    assert linear_nodes[0].get("id") not in names, model
                    assert all(node.get("id") in names for node in linear_nodes[1:]), model
                if row["committed"] == "1":
                    assert not np.array_equal(parameters, previous), model
                elif restart is None:
                    np.testing.assert_array_equal(parameters, previous, err_msg=f"rejected model: {model}")
                previous = parameters
                if orbital_model:
                    orbital_history.append(check_orbital_frame(group, tree, parameters, row, model))
                pseudo = tree.find("hamiltonian/pairpot/pseudo")
                if pseudo is not None:
                    physical = text_value(group["physical_definition"][()])
                    assert Path(pseudo.get("href")).read_bytes().decode() in physical, model
                    assert "rule=2;local_channel=1;" in physical and "knots=4;" in physical, model
                    assert pseudo.get("href") in text_value(group["ecp_source_report"][()]), model
                serialized = group["wavefunction_xml"][()]
                if isinstance(serialized, np.ndarray):
                    serialized = serialized.reshape(-1)[0]
                if isinstance(serialized, bytes):
                    serialized = serialized.decode()
                exported = ET.fromstring(serialized)
                coefficients = exported.find(".//sposet/coefficient")
                assert coefficients.attrib == fixed.attrib, model
                assert coefficients.text.split() == fixed.text.split(), model
                if orbital_model:
                    assert exported.find(".//rotated_sposet").attrib == tree.find(".//rotated_sposet").attrib
                    assert not exported.findall(".//sposet[@optimize='yes']"), model
                else:
                    assert all(node.get("optimize") == "no" for node in exported.findall(".//sposet")), model
                if linear_nodes:
                    exported_list = exported.find(".//multideterminant/detlist")
                    assert list(exported_list)[0].attrib == linear_nodes[0].attrib, model
        if len(orbital_history) == 2:
            first, second = orbital_history[0][0], orbital_history[1][0]
            delta = second - first
            assert first[0] * delta[1] != first[1] * delta[0], (directory, "fixture updates commute")
            assert orbital_history[-1][1][2] != 0, (directory, "missing accumulated virtual rotation")
        return directory

    if args.family_screen:
        reports = []
        for optimizer in (args.family_method or ("lm", "alm", "pii", "minpii")):
            def controls(tree):
                qmc = tree.find("qmc")
                qmc.find("parameter[@name='walkers_per_rank']").text = "4"
                qmc.find("parameter[@name='blocks']").text = "8"
                training = qmc.find("training")
                limits = {"damping": "0.01", "krylov_maximum_iterations": "64",
                          "krylov_maximum_operator_applications": "256", "krylov_maximum_seconds": "15",
                          "krylov_relative_tolerance": "1e-6"}
                if optimizer in ("lm", "alm"):
                    limits.update(lm_maximum_subspace_dimension="9")
                    if optimizer == "alm":
                        limits.update(lm_inner_iterations="32", lm_inner_restart_length="9")
                else:
                    limits.update(gmres_restart_length="16", maximum_damping_retries="0",
                                  pii_energy_undershoot="1.0")
                for name, value in limits.items():
                    ET.SubElement(training, "parameter", name=name).text = value
            try:
                directory = run(optimizer, optimizer=optimizer, mutate=controls,
                                samples_per_rank=32, require_committed=False)
                rows = history(directory)
                decisions = ",".join(row["committed"] for row in rows)
                reports.append((optimizer, "complete", decisions))
                print(f"{optimizer}: complete; committed={decisions}", flush=True)
            except (AssertionError, subprocess.TimeoutExpired) as error:
                reports.append((optimizer, "failed", ""))
                print(f"{optimizer}: FAILED; {error}", flush=True)
        with (work / "family_results.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("optimizer", "status", "committed"))
            writer.writerows(reports)
        assert all(status == "complete" for _, status, _ in reports), work
        print("PASS: bounded LM/ALM/PII/minPII physical-state consistency; decisions recorded without forcing acceptance")
        return

    if args.optimizer_screen:
        for optimizer in ("momentum_sgd", "rmsprop", "amsgrad", "sr", "minsr", "spring"):
            def controls(tree):
                if optimizer in ("sr", "minsr", "spring"):
                    training = tree.find("qmc/training")
                    limits = {"damping": "0.01", "krylov_maximum_iterations": "32",
                              "krylov_maximum_operator_applications": "128", "krylov_maximum_seconds": "10",
                              "krylov_relative_tolerance": "1e-8"}
                    if optimizer == "sr":
                        limits["maximum_damping_attempts"] = "1"
                    for name, value in limits.items():
                        ET.SubElement(training, "parameter", name=name).text = value
            directory = run(optimizer, optimizer=optimizer, mutate=controls)
            for row in history(directory):
                for name, maximum in (("krylov_iterations", 32), ("krylov_operator_applications", 128)):
                    if row.get(name):
                        assert int(row[name]) <= maximum, (optimizer, row)
        print("PASS: bounded momentum_sgd/rmsprop/amsgrad/sr/minsr/spring SJ updates and inference")
        return

    adam = run("adam")
    if args.model_import:
        exported = sorted((adam / "sj.s000.training").rglob("*.parameters.h5"))[-1]
        imported_vmc = run("imported_vmc", vmc=True, parameter_file=exported)
        explicit_vmc = run("explicit_vmc", vmc=True, reference_file=exported)
        np.testing.assert_array_equal(vmc_physical_trace(imported_vmc), vmc_physical_trace(explicit_vmc))
        trained = run("imported_training", parameter_file=exported)
        split = run("imported_split", iterations=1, parameter_file=exported)
        resumed = run("imported_resumed", parameter_file=exported,
                      restart=split / "sj.s000.last_checkpoint.h5")
        final = sorted((trained / "sj.s000.training").rglob("*.parameters.h5"))[-1]
        restored = sorted((resumed / "sj.s000.training").rglob("*.parameters.h5"))[-1]
        lhs, rhs = datasets(final), datasets(restored)
        assert lhs.keys() == rhs.keys()
        for name in lhs:
            np.testing.assert_array_equal(lhs[name], rhs[name], err_msg=f"initialized restart: {name}")
        # A second import still uses ORIGINAL inline XML and original registration
        # identity, rather than a rewritten XML whose baseline masks a loader bug.
        second_vmc = run("reimported_vmc", vmc=True, parameter_file=final)
        second_reference = run("reexport_reference_vmc", vmc=True, reference_file=final)
        np.testing.assert_array_equal(vmc_physical_trace(second_vmc), vmc_physical_trace(second_reference))
        for name in ("model_identity", "registration_values", "wavefunction_xml", "physical_definition"):
            np.testing.assert_array_equal(datasets(exported)["conventional_wavefunction/" + name],
                                          lhs["conventional_wavefunction/" + name])
        missing = work / "missing-parameters.h5"
        shutil.copyfile(exported, missing)
        with h5py.File(missing, "r+") as handle:
            del handle["conventional_wavefunction/parameters"]
        run("reject_missing_import_parameters", parameter_file=missing, vmc=True,
            reject="Conventional model export")
        run("reject_unsupported_import_driver", parameter_file=exported, vmc=True,
            mutate=lambda tree: tree.find("qmc").set("method", "dmc"),
            reject="parameter_file supports only CPU batched VMC and WFTrain")
        if ET.parse(fixture).find("hamiltonian/pairpot/pseudo") is not None:
            source = (fixture.parent / ET.parse(fixture).find("hamiltonian/pairpot/pseudo").get("href")).resolve()
            original = source.read_bytes()
            changed = work / "changed-import-pseudopotential.xml"
            changed.write_bytes(original + b"\n<!-- imported model PP identity regression -->\n")
            run("reject_changed_import_pp", parameter_file=exported, vmc=True,
                mutate=lambda tree: tree.find("hamiltonian/pairpot/pseudo").set("href", str(changed)),
                reject="Conventional model export identity mismatch")
            assert source.read_bytes() == original
        print("PASS: conventional model import physical VMC reference, fresh training/re-export/re-import, "
              "exact initialized split restart and malformed/unsupported import rejection")
        return
    if args.profile_only:
        print(f"PASS: bounded conventional Adam/inference {args.ranks} MPI x {args.threads} OpenMP; accepted model export")
        return
    off = run("adam_no_inference", inference=False)
    def companion_controls(tree):
        if args.companion_optimizer == "sr":
            limits = {"damping": "0.01", "krylov_maximum_iterations": "32",
                      "krylov_maximum_operator_applications": "128", "krylov_maximum_seconds": "10",
                      "krylov_relative_tolerance": "1e-8", "maximum_damping_attempts": "1"}
            for name, value in limits.items():
                ET.SubElement(tree.find("qmc/training"), "parameter", name=name).text = value
    run(args.companion_optimizer, optimizer=args.companion_optimizer, mutate=companion_controls)
    for lhs, rhs in zip(history(adam), history(off)):
        for name in ("iteration", "samples", "mean_energy", "variance", "gradient_norm", "selected_alpha", "committed"):
            assert lhs[name] == rhs[name], (name, lhs[name], rhs[name])
    for pattern in ("*.parameters.h5", "*.state.h5", "*.rank_*.h5"):
        compare_archives(adam / "sj.s000.training", off / "sj.s000.training", pattern)
    if args.restart:
        split = run("split", iterations=1)
        resumed = run("resumed", restart=split / "sj.s000.last_checkpoint.h5")
        for name in ("samples", "mean_energy", "variance", "gradient_norm", "selected_alpha", "committed"):
            assert history(adam)[-1][name] == history(resumed)[0][name], name
        # Compare the final model rather than wall-time/counter metadata.
        final = sorted((adam / "sj.s000.training").rglob("*.parameters.h5"))[-1]
        restored = sorted((resumed / "sj.s000.training").rglob("*.parameters.h5"))[-1]
        lhs, rhs = datasets(final), datasets(restored)
        assert lhs.keys() == rhs.keys()
        for name in lhs:
            np.testing.assert_array_equal(lhs[name], rhs[name], err_msg=f"restart model: {name}")
    if args.preflight:
        run("reject_kfac", optimizer="kfac", reject="no affine observations required by KFAC")
        run("reject_zero_cap", reject="legacy_maximum_parameters must be positive", mutate=lambda tree: setattr(
            tree.find("qmc/training/parameter[@name='legacy_maximum_parameters']"), "text", "0"))

        def neural_decay(tree):
            ET.SubElement(tree.find("qmc/training"), "parameter", name="weight_decay").text = "0.01"
        run("reject_neural_decay", reject="weight_decay must be zero", mutate=neural_decay)
        if ET.parse(fixture).find("hamiltonian/pairpot/pseudo") is not None:
            ordinary_vp_reason = "ECP training requires ordinary locality with virtual-particle batching"
            run("reject_dla", reject=ordinary_vp_reason, mutate=lambda tree: tree.find(
                "hamiltonian/pairpot[@type='pseudo']").set("DLA", "yes"))
            run("reject_non_vp", reject=ordinary_vp_reason, mutate=lambda tree: tree.find(
                "hamiltonian/pairpot[@type='pseudo']").set("algorithm", "non-batched"))
            # Same scientific radial data with a harmless comment must still
            # reject exact continuation: identity is actual parsed content, not
            # the old filename or just the species/header.
            source = ET.parse(fixture).find("hamiltonian/pairpot/pseudo").get("href")
            source = (fixture.parent / source).resolve()
            original = source.read_bytes()
            changed = work / "changed-pseudopotential.xml"
            changed.write_bytes(original + b"\n<!-- exact PP identity regression -->\n")
            run("reject_changed_pp", iterations=3, restart=adam / "sj.s000.last_checkpoint.h5",
                reject="Training checkpoint stage identity or configuration mismatch",
                mutate=lambda tree: tree.find("hamiltonian/pairpot/pseudo").set("href", str(changed)))
            assert source.read_bytes() == original
    completed = f"Adam/{args.companion_optimizer} conventional updates, interval inference, immutable model reference, exact inference isolation"
    if args.restart:
        completed += ", exact split restart"
    if args.preflight:
        completed += ", expected preflight diagnostics"
    print("PASS: " + completed)


if __name__ == "__main__":
    main()
