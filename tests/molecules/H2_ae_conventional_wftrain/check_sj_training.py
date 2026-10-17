#!/usr/bin/env python3
"""Bounded native SJ update/inference regression; retain all evidence in a temp dir."""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import re
from pathlib import Path
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def datasets(path):
    result = {}
    with h5py.File(path, "r") as handle:
        handle.visititems(lambda name, obj: result.update({name: obj[()]})
                          if isinstance(obj, h5py.Dataset) else None)
    return result


def compare_archives(left, right, pattern, *, optimizer_max_ulps=0):
    first = sorted(left.rglob(pattern))
    second = sorted(right.rglob(pattern))
    assert first and len(first) == len(second), (pattern, first, second)
    for a, b in zip(first, second):
        lhs, rhs = datasets(a), datasets(b)
        assert lhs.keys() == rhs.keys(), (a, b)
        moment_names = {"wftrain_checkpoint/optimizer/first_moment", "wftrain_checkpoint/optimizer/second_moment"}
        moment_difference = False
        if optimizer_max_ulps:
            # MD-only cold-process legacy roundoff: callers must first actually
            # restore BOTH own bundles, validating each fingerprint/payload.
            assert pattern == "*.state.h5" and optimizer_max_ulps == 8
            assert moment_names <= lhs.keys()
            for name in sorted(moment_names):
                first_moment, second_moment = np.asarray(lhs[name]), np.asarray(rhs[name])
                assert first_moment.dtype == second_moment.dtype == np.dtype("float64")
                assert first_moment.shape == second_moment.shape
                assert np.isfinite(first_moment).all() and np.isfinite(second_moment).all()
                distance = np.testing.assert_array_max_ulp(first_moment, second_moment, maxulp=optimizer_max_ulps)
                maximum = int(np.max(distance, initial=0))
                moment_difference |= maximum != 0
                print(f"MD-only legacy cold-process roundoff: {name} max_ULP={maximum} cap={optimizer_max_ulps}", flush=True)
        for name in lhs:
            if optimizer_max_ulps and (name in moment_names or
                    (moment_difference and name == "wftrain_checkpoint/optimizer/state_fingerprint")):
                continue
            np.testing.assert_array_equal(lhs[name], rhs[name], err_msg=f"{pattern}: {name}")


def history(directory):
    with (directory / "sj.s000.training_info.csv").open() as handle:
        return list(csv.DictReader(handle))


def text_value(value):
    value = np.asarray(value).reshape(-1)[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def four_mo_md_pairs(tree):
    """Derive expected local pairs from explicit DETS or CSF child occupations."""
    expansion = tree.find(".//multideterminant/detlist")
    csf = expansion.get("type") == "CSF"
    assert csf or expansion.get("type") == "DETS"
    determinants = [child for outer in expansion for child in outer] if csf else list(expansion)
    result = []
    for spin in range(2):
        occupations = [node.get(("alpha", "beta")[spin] if csf else f"occ{spin}") for node in determinants]
        assert all(len(value) == 4 for value in occupations)
        core = {i for i in range(4) if all(value[i] == "1" for value in occupations)}
        virtual = {i for i in range(4) if all(value[i] == "0" for value in occupations)}
        result.append([(i, j) for i in range(4) for j in range(i + 1, 4)
                       if not ({i, j} <= core or {i, j} <= virtual)])
    return result


def check_orbital_frame(group, tree, parameters, row, model, external_reference=None):
    """Check qualified frames against explicit inline or captured HDF references."""
    wrappers = tree.findall(".//rotated_sposet")
    multideterminant = tree.find(".//multideterminant") is not None
    assert len(wrappers) == (2 if multideterminant else 1)
    many_occupied = wrappers[0].find("sposet").get("size") == "4"
    pairs = (((0, 2), (0, 3), (1, 2), (1, 3)) if many_occupied else
             ((0, 1), (0, 2), (1, 2)) if multideterminant else ((0, 1), (0, 2)))
    owner_pairs = four_mo_md_pairs(tree) if many_occupied and multideterminant else [pairs] * len(wrappers)
    expected = [f"{wrapper.get('name')}_orb_rot_{i:04d}_{j:04d}_r"
                for wrapper, pairs in zip(wrappers, owner_pairs) for i, j in pairs]
    names = [text_value(value) for value in np.asarray(group["variable_names"][()]).reshape(-1)]
    indices = np.asarray(group["active_indices"][()]).reshape(-1)
    assert sorted(name for name in names if "_orb_rot_" in name) == sorted(expected), (model, names)
    active = parameters[[int(indices[names.index(name)]) for name in expected]]
    frame = group["orbital_frame"]
    np.testing.assert_array_equal(frame["format_version"][()], [1, 0, 0])
    np.testing.assert_array_equal(frame["determinant_aliases"][()], [0, 1] if multideterminant else [0, 0])
    assert int(np.asarray(frame["source_version"][()]).reshape(-1)[0]) == int(row["parameter_version"])
    assert int(np.asarray(frame["orbital_count"][()]).reshape(-1)[0]) == len(wrappers)
    full_coordinates = []
    for ordinal, wrapper in enumerate(wrappers):
        size = int(wrapper.find("sposet").get("size"))
        assert size == (4 if many_occupied else 3)
        orbital = frame[f"spo_{ordinal}"]
        assert text_value(orbital["name"][()]) == wrapper.get("name")
        initial = (external_reference.reshape(-1) if external_reference is not None else
                   np.asarray([float(value) for value in wrapper.find("sposet/coefficient").text.split()]))
        assert (initial.size == 216 if external_reference is not None else
                initial.size == 16 if many_occupied else initial.size in (9, 12))
        np.testing.assert_array_equal(orbital["shape"][()], [size, initial.size // size])
        reference = np.asarray(orbital["reference_coefficients"][()]).reshape(-1)
        accepted = np.asarray(orbital["accepted_coefficients"][()]).reshape(-1)
        full = np.asarray(orbital["full_rotation"][()]).reshape(-1)
        np.testing.assert_array_equal(reference, initial)
        assert accepted.size == initial.size and full.size == size * (size - 1) // 2
        assert np.isfinite(accepted).all() and np.isfinite(full).all()
        if row["committed"] == "1":
            assert not np.array_equal(accepted, reference), model
        full_coordinates.append(full)
    cookie = "conventional-rotated-lcao-mdj-v1" if multideterminant else "conventional-rotated-lcao-sj-v1"
    assert text_value(group["completion_cookie"][()]) == cookie
    # Both bundle representations must carry exactly the same complete frame.
    exported = datasets(model)
    iteration_root = model.name[:-len(".parameters.h5")]
    state = datasets(model.parent / (iteration_root + ".checkpoint") / (iteration_root + ".state.h5"))
    prefix = "conventional_wavefunction/orbital_frame/"
    for name, values in exported.items():
        if name.startswith(prefix):
            np.testing.assert_array_equal(values, state["wftrain_checkpoint/orbital_frame/" + name[len(prefix):]],
                                          err_msg=f"orbital checkpoint/export frame: {name}")
    return active, np.concatenate(full_coordinates)


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
        for ordinal, wrapper in enumerate(tree.findall(".//rotated_sposet")):
            # Remove the coordinate wrapper entirely: the independent VMC oracle
            # starts from exact accepted C, not an exp(active) reconstruction.
            spo = wrapper.find("sposet")
            spo.set("name", wrapper.get("name"))
            spo.find("coefficient").text = " ".join(str(value) for value in np.asarray(
                group[f"orbital_frame/spo_{ordinal}/accepted_coefficients"][()]).reshape(-1))
            collection = tree.find("wavefunction/sposet_collection")
            collection.remove(wrapper)
            collection.append(spo)


def apply_external_orbital_reference(tree, model, vp):
    """Use existing full-history VP restore, never inline-C text over an HDF reader."""
    with h5py.File(model, "r") as handle, h5py.File(vp, "w") as output:
        group = handle["conventional_wavefunction"]
        names = [text_value(value) for value in group["variable_names"][()].reshape(-1)]
        indices = np.asarray(group["active_indices"][()]).reshape(-1)
        parameters = np.asarray(group["parameters"][()]).reshape(-1)
        assert parameters.size == len(names) == 6 and sorted(indices) == list(range(6)) and np.isfinite(parameters).all()
        values = parameters[indices.astype(int)]
        active = dict(zip(names, values))
        name = tree.find(".//rotated_sposet").get("name")
        full = np.asarray(group["orbital_frame/spo_0/full_rotation"][()]).reshape(-1)
        assert full.size == 3 and np.isfinite(full).all()
        output["version"] = np.asarray([1, 1, 0], dtype=np.int32)
        lists = output.create_group("name_value_lists")
        # Match the legacy vector<string> proxy's variable-length H5T_C_S1 ASCII type.
        lists.create_dataset("parameter_names", data=names, dtype=h5py.string_dtype("ascii"))
        lists["parameter_values"] = values
        rotation = output.create_group("RotatedSPOs")
        rotation.create_group("rotation_global")[f"rotation_global_{name}"] = full
        rotation.create_group("rotation_params")[f"rotation_params_{name}"] = [
            active[f"{name}_orb_rot_0000_{q:04d}_r"] for q in (1, 2)]
    ET.SubElement(tree.find("wavefunction"), "override_variational_parameters", href=str(vp))


def apply_legacy_reference(tree, vp_file):
    """Map legacy full-precision VP coordinates into a separate XML oracle."""
    saved = datasets(vp_file)
    names = [text_value(name) for name in saved["name_value_lists/parameter_names"].reshape(-1)]
    values = saved["name_value_lists/parameter_values"].reshape(-1)
    active = dict(zip(names, values))
    for node in tree.findall("wavefunction/jastrow/correlation/coefficients"):
        node.text = " ".join(str(active[f"{node.get('id')}_{i}"]) for i in range(len(node.text.split())))


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


def prepare_periodic_spline_fixture(original, destination):
    """Derive the qualified diamond model, reusing existing tiny native controls."""
    tree = ET.parse(original)
    root = tree.getroot()
    system = root.find("qmcsystem")
    assert system is not None
    for node in list(system):
        root.append(node)
    root.remove(system)
    root.remove(root.find("qmc"))
    root.insert(1, ET.Element("random", seed="1931"))
    cell = root.find("simulationcell")
    cell.find("parameter[@name='LR_dim_cutoff']").text = "30"
    ET.SubElement(cell, "parameter", name="LR_handler").text = "ewald"
    # QMCMain reads simulationcell only through processPWH(qmcsystem).
    root.remove(cell)
    cell_system = ET.Element("qmcsystem")
    cell_system.append(cell)
    root.insert(2, cell_system)
    collection = root.find("wavefunction/sposet_collection")
    collection.set("href", str((original.parent / collection.get("href")).resolve()))
    collection.set("gpu", "no")
    collection.find("sposet").set("optimize", "no")
    assert collection.get("precision") == "double" and collection.find("sposet").get("size") == "4"
    ham = root.find("hamiltonian")
    for node in list(ham):
        if node.get("type") == "MPC" or node.tag == "estimator":
            ham.remove(node)
    pp = ham.find("pairpot/pseudo")
    pp.set("href", str((original.parent / pp.get("href")).resolve()))
    pp.set("format", "xml"); pp.set("nrule", "4"); pp.set("l-local", "1")
    pairpot = next(node for node in ham if pp in list(node))
    pairpot.set("algorithm", "batched"); pairpot.set("DLA", "no")
    root.append(ET.parse(Path(__file__).with_name("wftrain_sj.xml")).find("qmc"))
    tree.write(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("fixture", type=Path, nargs="?", default=Path(__file__).with_name("wftrain_sj.xml"))
    parser.add_argument("--restart", action="store_true", help="also require split-run exact continuation")
    parser.add_argument("--preflight", action="store_true", help="also check unsupported model/optimizer rejection")
    parser.add_argument("--model-import", action="store_true",
                        help="export/import ordinary VMC, fresh training, re-export and exact restart")
    parser.add_argument("--old-new", action="store_true",
                        help="bounded original batched OneShiftOnly versus new Adam integration comparison")
    parser.add_argument("--ci-old-new", action="store_true",
                        help="one H4 pure-CI OneShiftOnly/full-LM candidate pair; no equal-update claim")
    parser.add_argument("--ci-oracle", type=Path,
                        help="existing test_new_drivers for the --ci-old-new actual-candidate physical oracle")
    parser.add_argument("--periodic-heg", action="store_true",
                        help="bounded HEG14 Gamma Adam/SR, full model import/restart and periodic identity negatives")
    parser.add_argument("--md-orbital", action="store_true",
                        help="bounded distinct-spin MD orbital/CI/J2 Adam/SR and complete-frame model reuse")
    parser.add_argument("--external-lcao", action="store_true",
                        help="bounded LiH CPU external LCAO Adam, exact source identity and model reuse")
    parser.add_argument("--external-orbital-ecp", action="store_true",
                        help="derive shared three-MO external orbital/J2 ECP P6 with a full-history VP reference")
    parser.add_argument("--periodic-spline", action="store_true",
                        help="derive Gamma diamond fixed-spline/J1/J2 Ewald30 locality-ECP Adam P24")
    parser.add_argument("--orbital-ecp", action="store_true",
                        help="bounded one-occupied Be orbital/J2 ECP Adam with complete-frame model reuse")
    parser.add_argument("--many-occupied-orbital-ecp", action="store_true",
                        help="bounded neutral-C two-occupied/four-MO P8 orbital/J2 ECP Adam")
    parser.add_argument("--md-orbital-ecp", action="store_true",
                        help="bounded neutral-C fixed-CI two-owner MD orbital/J2 ECP Adam P16")
    parser.add_argument("--joint-ci-orbital-ecp", action="store_true",
                        help="derive joint P17 CI1/orbital/J2 ECP Adam from the fixed-CI XML")
    parser.add_argument("--joint-ci-orbital-ecp-minpii", action="store_true",
                        help="same joint P17 round trip with minPII only; strict initialized restart")
    parser.add_argument("--joint-ci-orbital-ecp-alm", action="store_true",
                        help="same joint P17 ALM round trip or honest initial untrusted-root refusal")
    parser.add_argument("--unequal-spin-orbital-ecp", action="store_true",
                        help="derive one charged C^- 3up/2down joint P16 orbital ECP case")
    parser.add_argument("--csf-orbital-ecp", action="store_true",
                        help="derive one signed-child CSF/orbital/J2 ECP Adam P17 case")
    parser.add_argument("--csf-orbital-ecp-alm", action="store_true",
                        help="same signed-child CSF P17 strict workflow with ALM only")
    parser.add_argument("--complex-periodic-heg", action="store_true",
                        help="CPU complex Gamma HEG14 real-J2 Adam, complete split restart; no model import")
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
    parser.add_argument("--ci-pii", action="store_true",
                        help="bounded pure-CI H4 PII/minPII updates and exported inference; freeze J2")
    parser.add_argument("--ci-lm", action="store_true",
                        help="bounded pure-DETS H4 LM/ALM retraction, exported inference and exact restart")
    parser.add_argument("--csf-lm", action="store_true",
                        help="two-DETS-per-CSF LM positive and sampled complex-root ALM refusal")
    parser.add_argument("--csf-pii", action="store_true",
                        help="two-DETS-per-CSF PII/minPII additive updates, inference and restart")
    parser.add_argument("--mdj-pii", action="store_true",
                        help="mixed DETS/J2 PII/minPII tangent updates, inference and restart")
    parser.add_argument("--mdj-alm", action="store_true",
                        help="one fixed-orbital H4 DETS/J2 AE ALM full model-import round trip")
    parser.add_argument("--mdj-ecp-alm", action="store_true",
                        help="one fixed-orbital Be DETS/J2 locality-ECP ALM workflow or honest pre-update root refusal")
    parser.add_argument("--csf-j2-ecp-lm", action="store_true",
                        help="one fixed-orbital Be signed-CSF/J2 locality-ECP full LM workflow")
    parser.add_argument("--sj-reference-line-search", action="store_true",
                        help="H2 SJ Adam reference search with existing five alphas and natural decisions")
    parser.add_argument("--sj-endpoints-line-search", action="store_true",
                        help="same H2 SJ workflow with private endpoint warmup/sampling1 and natural decisions")
    parser.add_argument("--csf-j2-pii", action="store_true",
                        help="nontrivial mixed CSF/J2 PII/minPII updates, inference and restart")
    args = parser.parse_args()
    if args.ci_oracle and not args.ci_old_new:
        parser.error("--ci-oracle requires --ci-old-new")
    if args.ci_old_new:
        if args.ranks != 1 or args.threads != 1 or args.ci_representation != "dets" or any((
                args.old_new, args.restart, args.preflight, args.model_import, args.profile_only,
                args.optimizer_screen, args.family_screen, args.family_method, args.ci_pii, args.ci_lm,
                args.csf_lm, args.csf_pii, args.mdj_pii, args.mdj_alm, args.mdj_ecp_alm,
                args.csf_j2_ecp_lm, args.csf_j2_pii, args.orbital_jastrow, args.periodic_heg,
                args.complex_periodic_heg, args.periodic_spline, args.md_orbital, args.external_lcao,
                args.external_orbital_ecp, args.orbital_ecp, args.many_occupied_orbital_ecp,
                args.md_orbital_ecp, args.joint_ci_orbital_ecp, args.joint_ci_orbital_ecp_minpii,
                args.joint_ci_orbital_ecp_alm, args.unequal_spin_orbital_ecp, args.csf_orbital_ecp,
                args.csf_orbital_ecp_alm, args.sj_reference_line_search, args.sj_endpoints_line_search)):
            parser.error("--ci-old-new selects its own one-rank pure-DETS candidate comparison")
        args.old_new = True  # Reuse only existing bounded process/resource bookkeeping.
    if args.csf_orbital_ecp_alm:
        if args.ranks != 1 or args.threads != 1 or args.ci_representation != "dets" or any((
                args.csf_orbital_ecp, args.joint_ci_orbital_ecp, args.joint_ci_orbital_ecp_alm,
                args.joint_ci_orbital_ecp_minpii, args.md_orbital_ecp, args.unequal_spin_orbital_ecp,
                args.mdj_alm, args.mdj_ecp_alm)):
            parser.error("--csf-orbital-ecp-alm selects one single-rank signed-child CSF recipe")
        args.csf_orbital_ecp = True
    orbital_alm = args.joint_ci_orbital_ecp_alm or args.csf_orbital_ecp_alm
    if args.csf_j2_ecp_lm:
        if args.ranks != 1 or args.threads != 1 or args.ci_representation != "dets" or any((
                args.mdj_alm, args.mdj_ecp_alm, args.joint_ci_orbital_ecp_alm,
                args.csf_orbital_ecp_alm)):
            parser.error("--csf-j2-ecp-lm selects one single-rank Be CSF/J2 LM recipe")
        args.mdj_ecp_alm = True  # Reuse strict P5 model/import/restart checks, not its optimizer.
    if args.mdj_ecp_alm:
        if args.mdj_alm:
            parser.error("select only one AE or locality-ECP mixed ALM profile")
        args.mdj_alm = True  # Reuse strict initialized import/restart, not a second solver workflow.
    if args.sj_endpoints_line_search:
        if args.sj_reference_line_search:
            parser.error("select only one SJ line-search mode")
        args.sj_reference_line_search = True  # Reuse the strict fixed H2 search workflow below.
    if args.sj_reference_line_search:
        if args.ranks != 1 or args.threads != 1 or args.ci_representation != "dets" or any((
                args.restart, args.preflight, args.model_import, args.old_new, args.family_screen,
                args.family_method, args.profile_only, args.optimizer_screen, args.ci_pii, args.ci_lm,
                args.csf_lm, args.csf_pii, args.mdj_pii, args.mdj_alm, args.csf_j2_pii, args.periodic_heg,
                args.complex_periodic_heg, args.md_orbital, args.external_lcao, args.external_orbital_ecp,
                args.orbital_ecp, args.many_occupied_orbital_ecp, args.md_orbital_ecp,
                args.joint_ci_orbital_ecp, args.joint_ci_orbital_ecp_minpii,
                args.unequal_spin_orbital_ecp, args.csf_orbital_ecp, args.orbital_jastrow)):
            parser.error("--sj-reference-line-search selects one single-rank H2 SJ AE workflow")
        args.model_import = True
    if args.mdj_alm:
        if args.ranks not in (1, 2) or args.threads != 1 or args.ci_representation != "dets" or any((
                args.restart, args.preflight, args.model_import, args.old_new, args.family_screen,
                args.family_method, args.profile_only, args.optimizer_screen, args.ci_pii, args.ci_lm,
                args.csf_lm, args.csf_pii, args.mdj_pii, args.csf_j2_pii, args.periodic_heg,
                args.complex_periodic_heg, args.md_orbital, args.external_lcao, args.external_orbital_ecp,
                args.orbital_ecp, args.many_occupied_orbital_ecp, args.md_orbital_ecp,
                args.joint_ci_orbital_ecp, args.joint_ci_orbital_ecp_minpii,
                args.unequal_spin_orbital_ecp, args.csf_orbital_ecp, args.orbital_jastrow)):
            parser.error("mixed ALM selects one 1/2-rank fixed-orbital DETS/J2 workflow")
        args.model_import = True
    if args.joint_ci_orbital_ecp_minpii:
        if any((args.joint_ci_orbital_ecp, args.md_orbital_ecp, args.unequal_spin_orbital_ecp,
                args.csf_orbital_ecp)) or args.ci_representation != "dets":
            parser.error("--joint-ci-orbital-ecp-minpii selects its own equal-spin DETS recipe")
        args.joint_ci_orbital_ecp = True
    if args.joint_ci_orbital_ecp_alm:
        if any((args.joint_ci_orbital_ecp, args.md_orbital_ecp, args.unequal_spin_orbital_ecp,
                args.csf_orbital_ecp, args.mdj_alm)) or args.ci_representation != "dets":
            parser.error("--joint-ci-orbital-ecp-alm selects its own equal-spin DETS recipe")
        args.joint_ci_orbital_ecp = True
    if args.external_orbital_ecp:
        if args.external_lcao:
            parser.error("select fixed --external-lcao or active --external-orbital-ecp")
        args.external_lcao = True
    if args.csf_orbital_ecp:
        if args.md_orbital_ecp or args.joint_ci_orbital_ecp or args.unequal_spin_orbital_ecp or args.ci_representation != "dets":
            parser.error("--csf-orbital-ecp generates its own single CSF recipe; omit other representation modes")
        args.joint_ci_orbital_ecp = True
    if args.unequal_spin_orbital_ecp:
        if args.md_orbital_ecp or args.joint_ci_orbital_ecp:
            parser.error("select only one fixed, joint, or unequal-spin orbital ECP mode")
        args.joint_ci_orbital_ecp = True
    if args.joint_ci_orbital_ecp:
        if args.md_orbital_ecp:
            parser.error("select either fixed --md-orbital-ecp or --joint-ci-orbital-ecp")
        args.md_orbital_ecp = True
    orbital_ecp_profile = args.orbital_ecp or args.many_occupied_orbital_ecp or args.md_orbital_ecp
    periodic_profile = args.periodic_heg or args.complex_periodic_heg
    if args.complex_periodic_heg:
        assert args.threads == 1 and args.ranks > 0
        assert args.ci_representation == "dets"
        assert not any((args.periodic_heg, args.md_orbital, args.external_lcao, orbital_ecp_profile,
                        args.orbital_jastrow, args.old_new, args.optimizer_screen,
                        args.family_screen, args.profile_only, args.ci_pii, args.ci_lm, args.csf_lm,
                        args.csf_pii, args.mdj_pii, args.csf_j2_pii))
        if not args.model_import:
            args.restart = args.preflight = True
    linear_ci = args.ci_lm or args.csf_lm
    csf_ci = args.csf_lm or args.csf_pii or args.csf_j2_pii
    pure_ci = args.ci_pii or args.csf_pii or linear_ci
    mixed_ci = args.mdj_pii or args.csf_j2_pii
    ci_workflow = pure_ci or mixed_ci
    if ci_workflow:
        if args.ranks != 1 or args.threads != 1 or any((args.family_screen, args.family_method,
                args.profile_only, args.optimizer_screen, args.model_import, args.restart, args.preflight,
                args.old_new, args.md_orbital, args.periodic_heg, args.external_lcao, orbital_ecp_profile,
                args.orbital_jastrow)) or args.ci_representation != "dets" or sum((args.ci_pii, args.ci_lm, args.csf_lm, args.csf_pii, args.mdj_pii, args.csf_j2_pii)) != 1:
            parser.error("select one single-rank --ci-pii/--ci-lm/--csf-lm/--csf-pii/--mdj-pii/--csf-j2-pii workflow")
        args.family_screen = True
        args.family_method = ("lm", "alm") if linear_ci else ("pii", "minpii")
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
    if args.old_new and (args.ranks != 1 or any((args.model_import, args.profile_only, args.optimizer_screen,
                                               args.family_screen, args.restart, args.preflight))):
        parser.error("--old-new selects its own single-rank bounded workflow")
    executable = args.executable.resolve()
    fixture = args.fixture.resolve()
    work = Path(tempfile.mkdtemp(prefix="conventional_sj_verify_"))
    print(work, flush=True)
    spline_image = None
    if args.periodic_spline:
        assert args.ranks == args.threads == 1 and args.ci_representation == "dets"
        assert not any((args.periodic_heg, args.complex_periodic_heg, args.external_lcao,
                        args.external_orbital_ecp, args.md_orbital, orbital_ecp_profile, args.ci_pii,
                        args.ci_lm, args.csf_lm, args.csf_pii, args.mdj_pii, args.csf_j2_pii, args.mdj_alm,
                        args.sj_reference_line_search, args.old_new, args.family_screen, args.family_method,
                        args.optimizer_screen, args.profile_only, args.restart, args.preflight))
        fixture = prepare_periodic_spline_fixture(fixture, work / "diamond-wftrain.xml")
        spline_source = Path(ET.parse(fixture).find("wavefunction/sposet_collection").get("href"))
        spline_image = spline_source.read_bytes()
        args.model_import = True
    if args.periodic_heg or args.md_orbital or args.external_lcao or orbital_ecp_profile:
        assert sum((args.periodic_heg, args.md_orbital, args.external_lcao, args.orbital_ecp,
                    args.many_occupied_orbital_ecp, args.md_orbital_ecp)) == 1, "select one bounded qualification profile"
        allowed_ranks = (1, 2) if args.joint_ci_orbital_ecp_alm else (1,)
        assert args.ranks in allowed_ranks and args.threads == 1, "bounded profile MPI/OpenMP layout"
        assert not any((args.old_new, args.optimizer_screen, args.family_screen, args.profile_only,
                        args.restart, args.preflight)), "--periodic-heg selects its own bounded workflow"
        args.model_import = True
    fixture_tree = ET.parse(fixture)
    if args.ci_old_new:
        assert not fixture_tree.findall(".//pseudo") and not fixture_tree.findall(".//rotated_sposet")
        expansion = fixture_tree.find(".//multideterminant/detlist")
        assert expansion is not None and expansion.get("type") == "DETS" and expansion.get("nstates") == "3"
        assert [node.attrib for node in expansion] == [dict(id="CI0", coeff="0.6", occ0="110", occ1="110"),
                                                       dict(id="CI1", coeff="0", occ0="101", occ1="101")]
        for node in fixture_tree.findall("wavefunction/jastrow/correlation/coefficients"):
            node.set("optimize", "no")
        qmc = fixture_tree.find("qmc")
        for name, value in (("walkers_per_rank", "4"), ("blocks", "8"),
                            ("warmupsteps", "2"), ("timestep", "0.05")):
            qmc.find(f"parameter[@name='{name}']").text = value
        for name, value in (("damping", "0.01"), ("krylov_maximum_iterations", "64"),
                            ("krylov_maximum_operator_applications", "256"), ("krylov_maximum_seconds", "15"),
                            ("krylov_relative_tolerance", "1e-6"), ("lm_maximum_subspace_dimension", "2")):
            ET.SubElement(qmc.find("training"), "parameter", name=name).text = value
        fixture = work / "h4-pure-ci.xml"
        fixture_tree.write(fixture)
    if args.sj_reference_line_search:
        assert not fixture_tree.findall(".//multideterminant") and not fixture_tree.findall(".//pseudo")
        assert not fixture_tree.findall(".//rotated_sposet")
        assert sum(len(node.text.split()) for node in fixture_tree.findall(".//jastrow/correlation/coefficients")) == 8
    if args.mdj_alm:
        assert fixture_tree.find(".//detlist").get("type") == "DETS"
        assert not fixture_tree.findall(".//rotated_sposet")
        assert bool(fixture_tree.findall(".//pseudo")) == args.mdj_ecp_alm
        assert [node.get("coeff") for node in fixture_tree.findall(".//detlist/ci")] == \
            (["0.6", "0.05"] if args.mdj_ecp_alm else ["0.6", "0"])
        if args.mdj_ecp_alm:
            assert [node.get("elementType") for node in fixture_tree.findall(".//pseudo")] == ["Be"]
            assert sum(len(node.text.split()) for node in fixture_tree.findall(
                ".//jastrow/correlation/coefficients")) == 4
    external_ecp = args.external_lcao and bool(fixture_tree.findall("hamiltonian/pairpot/pseudo"))
    if external_ecp:
        assert [node.get("size") for node in fixture_tree.findall("particleset[@name='e']/group")] == ["1", "1"]
        assert {node.get("elementType"): node.get("l-local") for node in
                fixture_tree.findall("hamiltonian/pairpot/pseudo")} == {"Li": "1", "H": "0"}
    md_ecp = (fixture_tree.find("wavefunction/determinantset/multideterminant") is not None and
              fixture_tree.find("hamiltonian/pairpot/pseudo") is not None)
    csf_ecp = md_ecp and args.ci_representation == "csf"
    j3_ecp = (fixture_tree.find("wavefunction/jastrow[@type='eeI']") is not None and
              fixture_tree.find("hamiltonian/pairpot/pseudo") is not None)
    if j3_ecp:
        assert not md_ecp and fixture_tree.find(".//rotated_sposet") is None
    deadline = time.monotonic() + 300 if args.old_new or args.sj_reference_line_search or periodic_profile or args.periodic_spline or args.md_orbital or args.external_lcao or orbital_ecp_profile or ci_workflow or md_ecp or j3_ecp else None
    external_asset = None
    external_reference = None
    if args.external_lcao:
        collection = ET.parse(fixture).find("wavefunction/sposet_collection")
        assert collection.get("source_identity") == "file_image"
        external_source = (fixture.parent / collection.get("href")).resolve()
        external_original = external_source.read_bytes()
        external_asset = work / "original-lcao.h5"
        shutil.copyfile(external_source, external_asset)
        if args.external_orbital_ecp:
            assert external_ecp
            with h5py.File(external_asset, "r") as source:
                assert source["Super_Twist/eigenset_0"].shape == (62, 72)
                external_reference = source["Super_Twist/eigenset_0"][:3, :]

    def compare_vmc_reference(imported, reference):
        """Tolerate only VP exp(log) scientific values, never discrete counts or restart state."""
        actual, expected = vmc_physical_trace(imported), vmc_physical_trace(reference)
        if not args.external_orbital_ecp:
            np.testing.assert_array_equal(actual, expected)
            return
        headers = [next(line.split()[1:] for line in (directory / "sj.s000.scalar.dat").read_text().splitlines()
                        if line.startswith("#")) for directory in (imported, reference)]
        assert headers[0] == headers[1] and actual.shape == expected.shape
        columns = [name for name in headers[0] if name not in ("BlockCPU", "index")]
        exact = [i for i, name in enumerate(columns) if name in ("BlockWeight", "AcceptRatio") or "Count" in name]
        assert "BlockWeight" in columns
        np.testing.assert_array_equal(actual[:, exact], expected[:, exact])
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12,
                                   err_msg="full-history VP exp(log) physical reference ONLY")
        print(f"VP physical reference max_abs_difference={np.max(np.abs(actual - expected)):.17g}; "
              "rtol=atol=3e-12; discrete fields exact", flush=True)

    def launch(directory, ci_oracle=False):
        environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1")
        command = [str(executable), "run.xml"]
        if ci_oracle:
            environment["TASK46_CI_CANDIDATE_FILE"] = str((work / "candidate_c1.txt").resolve())
            command = [str(args.ci_oracle.resolve(strict=True)), "[legacy_adapter][ci_family_oracle]"]
        if args.ranks > 1:
            command = [args.mpiexec, "-np", str(args.ranks)] + command
        if args.old_new:
            command = ["/usr/bin/time", "-v", "-o", "resource_time.txt"] + command
        child_cap = 90 if ci_oracle else 120 if args.joint_ci_orbital_ecp_alm and args.ranks == 2 else 180
        timeout = child_cap if deadline is None else min(child_cap, deadline - time.monotonic())
        assert timeout > 0, (directory, "whole-checker 300s cap reached")
        started = time.monotonic()
        with (directory / "run.log").open("w") as log:
            process = subprocess.Popen(command, cwd=directory, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # GNU time is a parent of QMCPACK: stop this explicitly owned
                # group, not just the wrapper while leaving the application live.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                last = (directory / "run.log").read_text()[-2000:]
                raise AssertionError((directory, "process/whole-checker time cap; last native output", last))
        result = subprocess.CompletedProcess(command, code)
        text = (directory / "run.log").read_text()
        if args.old_new:
            source = re.search(r"Last git commit:\s*(\S+)", text)
            (directory / "execution.json").write_text(json.dumps({
                "elapsed_seconds": time.monotonic() - started, "exit_code": result.returncode,
                "native_source": source.group(1) if source else "unreported", "command": command,
                "environment": {name: environment.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                "input_sha256": hashlib.sha256((work / "candidate_c1.txt" if ci_oracle else directory / "run.xml").read_bytes()).hexdigest(),
                "memory_scope": "resource_time.txt per-process maxRSS, not aggregate peak"
            }, indent=2) + "\n")
        return result, text

    def run(label, *, optimizer=None, inference=True, iterations=2, restart=None, mutate=None, reject=False,
            samples_per_rank=4, require_committed=True, parameter_file=None, vmc=False, reference_file=None,
            legacy_vp=None, legacy_reference=None, completed_restore=False, ci_representation=None):
        directory = work / label
        directory.mkdir()
        if optimizer is None:
            optimizer = "lm" if args.csf_j2_ecp_lm else "alm" if args.mdj_alm or orbital_alm else "minpii" if args.joint_ci_orbital_ecp_minpii else "adam"
        if args.sj_reference_line_search:
            require_committed = False
        if args.mdj_alm or orbital_alm:
            samples_per_rank = 32
        if periodic_profile:
            samples_per_rank = 8
        tree = ET.parse(fixture)
        if args.joint_ci_orbital_ecp:
            # The only model-input delta from the qualified fixed-CI fixture.
            expansion = tree.find(".//multideterminant")
            assert expansion is not None and expansion.get("optimize") == "no"
            expansion.set("optimize", "yes")
        if args.unequal_spin_orbital_ecp:
            up = tree.find("particleset[@name='e']/group[@name='u']")
            up.set("size", "3")
            up.find("attrib[@name='position']").text = ".2 -.3 .4  .6 .2 -.5  .4 -.7 .6"
            expansion = tree.find(".//multideterminant/detlist")
            expansion.set("ne0", "3")
            expansion[0].set("occ0", "1110")
            expansion[1].set("occ0", "1101")
        if args.csf_j2_ecp_lm:
            # Exact Ne2/two-MO physical model4: outer gauge is .6, not child-weighted .48.
            expansion = tree.find(".//multideterminant/detlist")
            expansion.set("type", "CSF")
            for old, new in (("nc0", "nca"), ("nc1", "ncb"), ("ne0", "nea"), ("ne1", "neb")):
                expansion.set(new, expansion.attrib.pop(old))
            for node in list(expansion):
                expansion.remove(node)
            expansion.extend(ET.fromstring('''<recipe>
              <csf id="CI0" coeff="0.6" occ="20" exctLvl="0"><det coeff="0.8" alpha="10" beta="10"/></csf>
              <csf id="CI1" coeff="0.05" occ="11" exctLvl="1">
                <det coeff="0.7071067811865475244" alpha="10" beta="01"/>
                <det coeff="-0.7071067811865475244" alpha="01" beta="10"/>
              </csf></recipe>'''))
            assert float(expansion[0].get("coeff")) * float(expansion[0][0].get("coeff")) == 0.48
        if args.csf_orbital_ecp:
            expansion = tree.find(".//multideterminant/detlist")
            expansion.set("type", "CSF"); expansion.set("cutoff", "0")
            for old, new in (("nc0", "nca"), ("nc1", "ncb"), ("ne0", "nea"), ("ne1", "neb")):
                expansion.set(new, expansion.attrib.pop(old))
            for node in list(expansion):
                expansion.remove(node)
            expansion.extend(ET.fromstring('''<recipe>
              <csf id="CI0" coeff="0.6" occ="2200" exctLvl="0"><det coeff="0.8" alpha="1100" beta="1100"/></csf>
              <csf id="CI1" coeff="0.1" occ="2110" exctLvl="1">
                <det coeff="0.7071067811865475244" alpha="1100" beta="1010"/>
                <det coeff="-0.7071067811865475244" alpha="1010" beta="1100"/>
              </csf></recipe>'''))
        if ci_workflow:
            expansion = tree.find("wavefunction/determinantset/multideterminant/detlist")
            assert expansion is not None and expansion.get("type") == "DETS" and len(expansion) == 2
            assert float(expansion[0].get("coeff")) != 0
            assert (md_ecp and (args.mdj_pii or args.ci_lm)) or float(expansion[1].get("coeff")) == 0
            if md_ecp:
                assert args.mdj_pii or args.ci_lm, "bounded ECP family profiles are DETS/J2 PII or pure-DETS LM/ALM"
            assert tree.find(".//rotated_sposet") is None
            if pure_ci:
                for node in tree.findall("wavefunction/jastrow/correlation/coefficients"):
                    node.set("optimize", "no")
            if csf_ci:
                expansion.set("type", "CSF")
                for old, new in (("nc0", "nca"), ("nc1", "ncb"), ("ne0", "nea"), ("ne1", "neb")):
                    expansion.set(new, expansion.attrib.pop(old))
                for node in list(expansion):
                    expansion.remove(node)
                for ordinal, (occupation, pairs) in enumerate((
                        ("211", (("110", "101"), ("101", "110"))),
                        ("121", (("110", "011"), ("011", "110"))))):
                    csf = ET.SubElement(expansion, "csf", id=f"CI{ordinal}",
                        coeff="0.6" if ordinal == 0 else "0", occ=occupation, exctLvl="1")
                    for alpha, beta in pairs:
                        ET.SubElement(csf, "det", coeff="0.7071067811865475244", alpha=alpha, beta=beta)
        if args.external_lcao:
            tree.find("wavefunction/sposet_collection").set("href", str(external_asset))
        if args.external_orbital_ecp:
            collection = tree.find("wavefunction/sposet_collection")
            spo = collection.find("sposet")
            assert spo.get("name") == "spo-ud" and spo.get("size") == "1"
            collection.remove(spo)
            spo.set("name", "base"); spo.set("size", "3")
            ET.SubElement(collection, "rotated_sposet", name="spo-ud").append(spo)
        # Recipes run in fresh evidence directories, not beside their PP asset.
        # Keep the tracked source untouched; scientific identity uses its bytes.
        for pseudo in tree.findall("hamiltonian/pairpot/pseudo"):
            href = Path(pseudo.get("href"))
            if not href.is_absolute():
                pseudo.set("href", str((fixture.parent / href).resolve()))
        orbital_model = tree.find(".//rotated_sposet") is not None
        if orbital_ecp_profile:
            assert orbital_model and (tree.find(".//multideterminant") is not None) == args.md_orbital_ecp
            coefficients = tree.find(".//rotated_sposet/sposet/coefficient")
            size, occupied = (4, 2) if args.many_occupied_orbital_ecp or args.md_orbital_ecp else (3, 1)
            assert coefficients.get("size") == str(size) and len(coefficients.text.split()) == 4 * size
            assert [group.get("size") for group in tree.findall("particleset[@name='e']/group")] == \
                   (["3", "2"] if args.unequal_spin_orbital_ecp else [str(occupied)] * 2)
            if args.many_occupied_orbital_ecp or args.md_orbital_ecp:
                np.testing.assert_array_equal(np.asarray(coefficients.text.split(), dtype=float).reshape(4, 4), np.eye(4))
                assert tree.find("hamiltonian/pairpot/pseudo").get("elementType") == "C"
            if args.md_orbital_ecp:
                expansion = tree.find(".//multideterminant")
                assert expansion.get("optimize") == ("yes" if args.joint_ci_orbital_ecp else "no")
                if args.csf_orbital_ecp:
                    csfs = expansion.find("detlist")
                    assert csfs.attrib == dict(size="2", type="CSF", nca="0", ncb="0", nea="2", neb="2", nstates="4", cutoff="0")
                    assert [node.attrib for node in csfs] == [dict(id="CI0", coeff="0.6", occ="2200", exctLvl="0"),
                                                             dict(id="CI1", coeff="0.1", occ="2110", exctLvl="1")]
                    assert [[child.attrib for child in outer] for outer in csfs] == [
                        [dict(coeff="0.8", alpha="1100", beta="1100")],
                        [dict(coeff="0.7071067811865475244", alpha="1100", beta="1010"),
                         dict(coeff="-0.7071067811865475244", alpha="1010", beta="1100")]]
                    assert float(csfs[0].get("coeff")) * float(csfs[0][0].get("coeff")) == 0.48
                else:
                    assert expansion.find("detlist").get("type") == "DETS"
                    assert [node.attrib for node in expansion.find("detlist")] == [
                        dict(id="CI0", coeff="1", occ0="1110" if args.unequal_spin_orbital_ecp else "1100", occ1="1100"),
                        dict(id="CI1", coeff="0.1", occ0="1101" if args.unequal_spin_orbital_ecp else "1010", occ1="1010")]
                assert [len(pairs) for pairs in four_mo_md_pairs(tree)] == ([5, 6] if args.unequal_spin_orbital_ecp else [6, 6])
                wrappers = tree.findall(".//rotated_sposet")
                assert [node.get("name") for node in wrappers] == ["rot", "rot_dn"]
                for wrapper in wrappers:
                    np.testing.assert_array_equal(np.asarray(wrapper.find("sposet/coefficient").text.split(), dtype=float).reshape(4, 4), np.eye(4))
        if args.orbital_jastrow:
            assert orbital_model and tree.find("wavefunction/jastrow") is None, fixture
            tree.find("wavefunction").append(ET.fromstring('''
                <jastrow name="J2" type="Two-Body" function="Bspline">
                  <correlation rcut="4" size="4" speciesA="u" speciesB="d">
                    <coefficients id="ud" type="Array" optimize="yes">0.12 -0.08 0.04 -0.01</coefficients>
                  </correlation>
                </jastrow>'''))
        if (ci_representation or args.ci_representation) == "csf":
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
        if args.sj_reference_line_search:
            training.find("parameter[@name='line_search']").text = "endpoints" if args.sj_endpoints_line_search else "reference"
            if args.sj_endpoints_line_search:
                for name in ("line_search_warmup_steps", "line_search_sampling_steps"):
                    ET.SubElement(training, "parameter", name=name).text = "1"
            # Twice uniform for four samples; .25 would reject every nonuniform trial.
            for name, value in (("line_search_maximum_weight", "0.5"),
                                ("line_search_minimum_ess_fraction", "0.5"), ("line_search_confidence", "2")):
                ET.SubElement(training, "parameter", name=name).text = value
        if args.mdj_alm or orbital_alm:
            # Complete augmented subspace: Be P5, H4 P9 or joint-orbital P17.
            qmc.find("parameter[@name='walkers_per_rank']").text = "4"
            qmc.find("parameter[@name='blocks']").text = "8"
            if optimizer in ("lm", "alm"):
                limits = {"damping": "0.01", "krylov_maximum_iterations": "64",
                          "krylov_maximum_operator_applications": "256", "krylov_maximum_seconds": "15",
                          "krylov_relative_tolerance": "1e-6", "lm_maximum_subspace_dimension":
                          "18" if orbital_alm else "6" if args.mdj_ecp_alm else "10"}
                if optimizer == "alm":
                    limits.update(lm_inner_iterations="32", lm_inner_restart_length="10")
                for name, value in limits.items():
                    ET.SubElement(training, "parameter", name=name).text = value
        if args.joint_ci_orbital_ecp_minpii and optimizer == "minpii":
            # Reuse bounded family controls without changing the physical sampler.
            limits = {"damping": "0.01", "krylov_maximum_iterations": "64",
                      "krylov_maximum_operator_applications": "256", "krylov_maximum_seconds": "15",
                      "krylov_relative_tolerance": "1e-6", "gmres_restart_length": "16",
                      "maximum_damping_retries": "0", "pii_energy_undershoot": "1.0"}
            for name, value in limits.items():
                ET.SubElement(training, "parameter", name=name).text = value
        training.find("parameter[@name='iterations']").text = str(iterations)
        if not inference:
            qmc.remove(qmc.find("inference"))
        if restart is not None:
            ET.SubElement(training, "parameter", name="restart_path").text = str(restart)
        if parameter_file is not None:
            tree.find("wavefunction").set("parameter_file", str(parameter_file.resolve()))
        if reference_file is not None:
            if args.external_orbital_ecp:
                assert vmc and parameter_file is None and restart is None
                apply_external_orbital_reference(tree, reference_file, directory / "reference.vp.h5")
            else:
                apply_model_reference(tree, reference_file)
        if legacy_vp is not None:
            ET.SubElement(tree.find("wavefunction"), "override_variational_parameters", href=str(legacy_vp))
        if legacy_reference is not None:
            apply_legacy_reference(tree, legacy_reference)
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
        result, text = launch(directory)
        if (args.mdj_ecp_alm or orbital_alm) and label == "alm" and result.returncode != 0 and \
                "Live LM solve failed: untrusted_root" in text:
            # A nonphysical sampled root is a protected refusal, never a tuned positive.
            csv_file = directory / "sj.s000.training_info.csv"
            assert not csv_file.exists() or not history(directory), directory
            assert not list(directory.rglob("*.h5")), directory
            assert "WFTrain refreshed parameter version" not in text and "WFTrain iteration 0 version" not in text
            definition = ET.parse(directory / "run.xml")
            if args.csf_orbital_ecp_alm:
                assert ET.tostring(definition.find(".//detlist")) == ET.tostring(tree.find(".//detlist"))
            else:
                assert [node.get("coeff") for node in definition.findall(".//detlist/ci")] == \
                    [node.get("coeff") for node in fixture_tree.findall(".//detlist/ci")]
            assert ET.tostring(definition.find("wavefunction/sposet_collection")) == \
                ET.tostring(fixture_tree.find("wavefunction/sposet_collection"))
            print("EXPECTED REFUSAL: selected mixed DETS/J2 or joint-orbital ALM untrusted root before update/publication; "
                  "no positive inference/import/restart qualification", flush=True)
            raise SystemExit(0)
        if reject:
            assert result.returncode != 0, (directory, text[-4000:])
            if isinstance(reject, tuple):
                assert label == "reject_changed_import_pp" or (external_ecp and label in
                    ("reject_changed_import_pp_Li", "reject_changed_import_pp_H")) or (args.complex_periodic_heg and label in
                    ("reject_changed_cell_import", "reject_changed_cutoff_import")) or \
                    (args.periodic_spline and label == "reject_changed_spline_cutoff_import"), \
                    "alternative diagnostics are limited to qualified PP/complex periodic identity rejection"
                assert any(message in text for message in reject), (directory, reject, text[-4000:])
            else:
                assert reject in text, (directory, reject, text[-4000:])
            assert not list(directory.glob("*.training/*/*.state.h5")), directory
            return directory
        assert result.returncode == 0, (directory, text[-6000:])
        if completed_restore:
            assert args.md_orbital and restart is not None and not vmc
            assert f"WFTrain restored {iterations} completed iteration(s)" in text, directory
            assert "0 committed update(s)" in text and "WFTrain iteration " not in text, directory
            assert not list(directory.glob("*.training/**/*.parameters.h5")), directory
            return directory
        if vmc:
            vmc_physical_trace(directory)
            assert not list(directory.glob("*.training")), directory
            return directory
        rows = history(directory)
        expected_rows = iterations if restart is None else 1
        assert len(rows) == expected_rows, (directory, rows)
        if args.sj_reference_line_search:
            receipts = re.findall(r"WFTrain candidate alpha (\S+) loss (\S+) loss_difference (\S+) "
                                  r"difference_error (\S+) ess (\S+) maximum_weight (\S+) valid ([01])", text)
            cursor = 0
            for ordinal, row in enumerate(rows):
                remaining = [tuple(map(float, receipt)) for receipt in receipts[cursor:]]
                assert len(remaining) >= 6, (directory, remaining)
                expanded = args.sj_endpoints_line_search and remaining[4][0] != 0
                count = (7 if expanded else 10) if args.sj_endpoints_line_search else 6
                candidates = remaining[:count]
                cursor += count
                assert all(math.isfinite(value) for candidate in candidates for value in candidate), directory
                assert all(0 < candidate[4] <= samples_per_rank and 0 < candidate[5] <= 1
                           for candidate in candidates), directory
                if args.sj_endpoints_line_search:
                    pattern = [0, 1, 0, 1, .5, 1.5, 2] if expanded else [0, 1, 0, 1, 0, 1, .125, .25, .5, .75]
                    assert [candidate[0] for candidate in candidates] == pattern, (directory, candidates)
                    gate = candidates[2][6] == candidates[3][6] == 1 and candidates[2][2] - 2*candidates[2][3] > 0
                    assert expanded == gate, (directory, candidates)
                    candidates = candidates[2:] if expanded else candidates[4:]
                else:
                    assert {candidate[0] for candidate in candidates} == {0, .125, .25, .5, .75, 1}, directory
                alpha = float(row["selected_alpha"])
                assert alpha in {candidate[0] for candidate in candidates} and row["committed"] == str(int(alpha > 0)), row
                baseline = candidates[0]
                assert alpha == 0 or any(candidate[0] == alpha and candidate[6] == 1 and
                    ((candidate[1] - baseline[1] + 2*(candidate[3] + baseline[3]) < 0) if expanded else
                     candidate[2] + 2*candidate[3] < 0) for candidate in candidates), row
                detail = f"{count} finite phase receipts" if args.sj_endpoints_line_search else "five nonzero candidates, finite receipts"
                mode = "endpoint" if args.sj_endpoints_line_search else "reference"
                print(f"{mode} search: {directory.name} iteration={row['iteration']} selected_alpha={alpha}; "
                      f"{detail}, natural decision", flush=True)
            assert cursor == len(receipts), (directory, receipts)
        for row in rows:
            if args.ci_old_new:
                assert row["optimizer"] == "lm", row
            if require_committed:
                assert row["committed"] == "1", (directory, row)
            assert row["committed"] in ("0", "1"), (directory, row)
            assert int(row["samples"]) == samples_per_rank * args.ranks, (directory, row)
            assert all(math.isfinite(float(row[name])) for name in ("mean_energy", "variance", "gradient_norm")), row
            if require_committed:
                assert float(row["gradient_norm"]) > 0, row
            if args.joint_ci_orbital_ecp_minpii:
                assert 0 < int(row["krylov_operator_applications"]) <= 256, row
                assert 0 <= int(row["krylov_iterations"]) <= 64, row
            if args.mdj_alm or orbital_alm:
                assert row["optimizer"] == ("lm" if args.csf_j2_ecp_lm else "alm"), row
                assert 0 < int(row["krylov_operator_applications"]) <= 256, row
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
        initial = ([] if args.md_orbital_ecp and not args.joint_ci_orbital_ecp else
                   [float(node.get("coeff")) for node in linear_nodes[1:]])
        if orbital_model:
            initial += [0.0] * (sum(map(len, four_mo_md_pairs(tree))) if args.md_orbital_ecp else
                               4 if args.many_occupied_orbital_ecp else 6 if linear_nodes else 2)
        initial += [float(value) for node in tree.findall("wavefunction/jastrow/correlation/coefficients")
                   if node.get("optimize") != "no"
                   for value in node.text.split()]
        previous = (np.asarray(initial) if parameter_file is None else
                    datasets(parameter_file)["conventional_wavefunction/parameters"].reshape(-1))
        fixed = tree.find(".//sposet/coefficient")
        orbital_history = []
        optimizer_keys = tuple("wftrain_checkpoint/optimizer/" + name for name in
                               ("first_moment", "second_moment", "accepted_update_count"))
        previous_optimizer = None
        if args.sj_reference_line_search and restart is not None:
            previous_optimizer = datasets(sorted((restart.parent / "sj.s000.training").rglob("*.state.h5"))[-1])
        for model, row in zip(models, rows):
            with h5py.File(model, "r") as handle:
                group = handle["conventional_wavefunction"]
                parameters = np.asarray(group["parameters"][()]).reshape(-1)
                assert parameters.size == len(initial) and np.isfinite(parameters).all(), model
                if j3_ecp:
                    assert parameters.size == 12, model
                    names = [text_value(name) for name in np.asarray(group["variable_names"][()]).reshape(-1)]
                    indices = np.asarray(group["active_indices"][()]).reshape(-1)
                    j3_indices = [int(index) for name, index in zip(names, indices) if name.startswith("j3Be_")]
                    assert len(j3_indices) == 8 and min(j3_indices) >= 0, model
                    if row["committed"] == "1":
                        assert not np.array_equal(parameters[j3_indices], previous[j3_indices]), (model, "J3 did not update")
                if linear_nodes:
                    assert parameters.size == len(initial)
                    if pure_ci or args.ci_old_new:
                        assert parameters.size == 1
                    else:
                        assert parameters.size > len(linear_nodes) - 1
                    names = [value.decode() if isinstance(value, bytes) else str(value)
                             for value in np.asarray(group["variable_names"][()]).reshape(-1)]
                    assert linear_nodes[0].get("id") not in names, model
                    if args.md_orbital_ecp and not args.joint_ci_orbital_ecp:
                        assert all(node.get("id") not in names for node in linear_nodes), model
                    else:
                        assert all(node.get("id") in names for node in linear_nodes[1:]), model
                if args.sj_reference_line_search and float(row["selected_alpha"]) == 0:
                    np.testing.assert_array_equal(parameters, previous,
                        err_msg="rejected reference search changed accepted parameters")
                if args.sj_reference_line_search:
                    bundle = model.parent / (model.name[:-len(".parameters.h5")] + ".checkpoint")
                    optimizer_state = datasets(next(bundle.glob("*.state.h5")))
                    if float(row["selected_alpha"]) == 0:
                        for key in optimizer_keys:
                            expected = previous_optimizer[key] if previous_optimizer is not None else np.zeros_like(optimizer_state[key])
                            np.testing.assert_array_equal(optimizer_state[key], expected, err_msg=f"rejected search: {key}")
                    previous_optimizer = optimizer_state
                if row["committed"] == "1" and not args.ci_old_new:
                    assert not np.array_equal(parameters, previous), model
                    if args.periodic_spline:
                        assert parameters.size == 24
                        assert not np.array_equal(parameters[:8], previous[:8]), (model, "J1 did not update")
                        assert not np.array_equal(parameters[8:], previous[8:]), (model, "J2 did not update")
                    if args.mdj_alm:
                        assert parameters.size == (5 if args.mdj_ecp_alm else 9) and names[0] == "CI1", model
                        assert parameters[0] != previous[0], (model, "CI1 did not update")
                        assert not np.array_equal(parameters[1:], previous[1:]), (model, "J2 did not update")
                    if args.external_orbital_ecp:
                        names = [text_value(value) for value in group["variable_names"][()].reshape(-1)]
                        indices = group["active_indices"][()].reshape(-1)
                        for prefix, count in (("spo-ud_orb_rot_", 2), ("ud_", 4)):
                            block = [int(index) for name, index in zip(names, indices) if name.startswith(prefix)]
                            assert len(block) == count and min(block) >= 0
                            assert not np.array_equal(parameters[block], previous[block]), (model, prefix)
                    if args.md_orbital_ecp:
                        assert parameters.size == (17 if args.joint_ci_orbital_ecp and not args.unequal_spin_orbital_ecp else 16)
                        indices = np.asarray(group["active_indices"][()]).reshape(-1)
                        if args.joint_ci_orbital_ecp:
                            assert int(indices[names.index("CI1")]) == 0
                            assert parameters[0] != previous[0], (model, "CI1 did not update")
                        for prefix, count in (("rot_orb_rot_", 5 if args.unequal_spin_orbital_ecp else 6),
                                              ("rot_dn_orb_rot_", 6), ("ud_", 4)):
                            block = [int(index) for name, index in zip(names, indices) if name.startswith(prefix)]
                            assert len(block) == count and min(block) >= 0, (model, prefix)
                            assert not np.array_equal(parameters[block], previous[block]), (model, prefix, "did not update")
                    if args.many_occupied_orbital_ecp:
                        assert parameters.size == 8
                        names = [text_value(value) for value in np.asarray(group["variable_names"][()]).reshape(-1)]
                        indices = np.asarray(group["active_indices"][()]).reshape(-1)
                        orbital_indices = [int(index) for name, index in zip(names, indices) if "_orb_rot_" in name]
                        j2_indices = [int(index) for name, index in zip(names, indices) if name.startswith("ud_")]
                        assert len(orbital_indices) == len(j2_indices) == 4 and min(orbital_indices + j2_indices) >= 0
                        assert not np.array_equal(parameters[orbital_indices], previous[orbital_indices]), (model, "orbital block did not update")
                        assert not np.array_equal(parameters[j2_indices], previous[j2_indices]), (model, "J2 block did not update")
                    if csf_ecp or (md_ecp and args.mdj_pii):
                        assert parameters.size == 5 and len(linear_nodes) == 2, model
                        active_indices = np.asarray(group["active_indices"][()]).reshape(-1)
                        ci_index = int(active_indices[names.index(linear_nodes[1].get("id"))])
                        assert ci_index >= 0 and parameters[ci_index] != previous[ci_index], (model, "CI/CSF did not update")
                        assert not np.array_equal(np.delete(parameters, ci_index), np.delete(previous, ci_index)), \
                            (model, "J2 did not update")
                elif row["committed"] == "0" and restart is None:
                    np.testing.assert_array_equal(parameters, previous, err_msg=f"rejected model: {model}")
                previous = parameters
                if orbital_model:
                    orbital_history.append(check_orbital_frame(group, tree, parameters, row, model, external_reference))
                for pseudo in tree.findall("hamiltonian/pairpot/pseudo"):
                    physical = text_value(group["physical_definition"][()])
                    assert Path(pseudo.get("href")).read_bytes().decode() in physical, model
                    pairpot = next(node for node in tree.findall("hamiltonian/pairpot") if pseudo in list(node))
                    species = pseudo.get("elementType")
                    descriptor = (f"species={species};source={pairpot.get('source')};"
                                  f"target={tree.find('hamiltonian').get('target')};format={pseudo.get('format')};"
                                  f"rule={pseudo.get('nrule')};local_channel={pseudo.get('l-local')};")
                    assert descriptor in physical, (model, descriptor)
                    if not external_ecp or species == "Li":
                        assert f"knots={12 if args.periodic_spline else 4};" in physical, model
                    assert f"species={species} href={pseudo.get('href')}" in text_value(group["ecp_source_report"][()]).splitlines(), model
                if periodic_profile:
                    physical = text_value(group["physical_definition"][()])
                    assert "handler=EwaldHandler3D;" in physical, model
                    assert "3d-uniform-neutralizing-background+self-v1" in physical, model
                    assert parameters.size == 10
                    if args.complex_periodic_heg:
                        assert parameters.dtype == np.dtype("float64"), model
                        assert "parameter=REAL64;wavefunction=COMPLEX128;" in physical, model
                        assert "complex_Gamma_FreeOrbital_kvectors_v1:" in physical, model
                if args.external_lcao:
                    if external_ecp:
                        assert parameters.size == (6 if args.external_orbital_ecp else 4) and not linear_nodes, model
                        assert orbital_model == args.external_orbital_ecp, model
                    image = external_asset.read_bytes()
                    physical = text_value(group["physical_definition"][()])
                    assert f"lcao_file_image_v1:{len(image)}:{image.hex()};" in physical, model
                    assert int(np.asarray(group["immutable_asset_bytes"][()]).reshape(-1)[0]) == len(image), model
                    assert text_value(group["lcao_source_report"][()]) == str(external_asset), model
                serialized = group["wavefunction_xml"][()]
                if isinstance(serialized, np.ndarray):
                    serialized = serialized.reshape(-1)[0]
                if isinstance(serialized, bytes):
                    serialized = serialized.decode()
                exported = ET.fromstring(serialized)
                if j3_ecp:
                    original_j3 = list(tree.find("wavefunction/jastrow[@type='eeI']").iter())
                    retained_j3 = list(exported.find("jastrow[@type='eeI']").iter())
                    assert len(original_j3) == len(retained_j3)
                    for original, retained in zip(original_j3, retained_j3):
                        assert (original.tag, original.attrib, (original.text or "").split()) == \
                               (retained.tag, retained.attrib, (retained.text or "").split()), model
                coefficients = exported.find(".//sposet/coefficient")
                if fixed is None:
                    assert (periodic_profile or args.periodic_spline) and coefficients is None, model
                    def fixed_spo_definition(node):
                        return [(child.tag, child.attrib, (child.text or "").split()) for child in node.iter()]
                    expected_spo = ET.fromstring(ET.tostring(tree.find("wavefunction/sposet_collection")))
                    if args.periodic_spline:
                        del expected_spo.attrib["href"]
                        physical = text_value(group["physical_definition"][()])
                        assert f"spline_source_v1:{len(spline_image)}:{spline_image.hex()};used_table:" in physical, model
                        assert re.search(r";used_table:[^;]+;", physical), model
                        assert int(np.asarray(group["immutable_asset_bytes"][()]).reshape(-1)[0]) == len(spline_image)
                        assert physical.count("handler=EwaldHandler3D;") == 3, model
                        assert physical.count(";lr_dim_cutoff=0x1.ep+4;") == 3, model
                        assert "periodic_local_pp:" in physical and "CoulombPBCAB:" in physical, model
                    assert fixed_spo_definition(exported.find("sposet_collection")) == fixed_spo_definition(
                        expected_spo), model
                else:
                    assert coefficients.attrib == fixed.attrib, model
                    assert (coefficients.text or "").split() == (fixed.text or "").split(), model
                if args.external_lcao:
                    captured = exported.find("sposet_collection")
                    assert "href" not in captured.attrib and captured.get("source_identity") == "file_image", model
                if orbital_model:
                    assert exported.find(".//rotated_sposet").attrib == tree.find(".//rotated_sposet").attrib
                    assert not exported.findall(".//sposet[@optimize='yes']"), model
                else:
                    assert all(node.get("optimize") == "no" for node in exported.findall(".//sposet")), model
                if linear_nodes:
                    exported_list = exported.find(".//multideterminant/detlist")
                    assert list(exported_list)[0].attrib == linear_nodes[0].attrib, model
                    if args.md_orbital_ecp or args.csf_j2_ecp_lm:
                        assert [node.attrib for node in exported_list] == [node.attrib for node in linear_nodes], model
                        if args.csf_orbital_ecp or args.csf_j2_ecp_lm:
                            assert [[child.attrib for child in outer] for outer in exported_list] == \
                                   [[child.attrib for child in outer] for outer in linear_nodes], model
                        if args.csf_j2_ecp_lm:
                            assert exported_list.get("type") == "CSF" and [len(outer) for outer in exported_list] == [1, 2]
                            assert float(exported_list[0].get("coeff")) == 0.6
                            assert float(exported_list[0][0].get("coeff")) == 0.8
                    if csf_ci or csf_ecp:
                        assert exported_list.get("type") == "CSF"
                        for original, retained in zip(linear_nodes, list(exported_list)):
                            assert len(original) == len(retained) == (2 if csf_ci else 1)
                            assert [det.attrib for det in original] == [det.attrib for det in retained], model
                if pure_ci or args.ci_old_new:
                    original_j2 = tree.findall("wavefunction/jastrow/correlation/coefficients")
                    retained_j2 = exported.findall("jastrow/correlation/coefficients")
                    assert len(original_j2) == len(retained_j2) == (1 if md_ecp else 2)
                    for original, retained in zip(original_j2, retained_j2):
                        assert retained.attrib == original.attrib and retained.text.split() == original.text.split(), model
        if len(orbital_history) == 2:
            first, second = orbital_history[0][0], orbital_history[1][0]
            delta = second - first
            if args.md_orbital_ecp:
                def generator(values, pairs):
                    matrix = np.zeros((4, 4))
                    for (i, j), value in zip(pairs, values):
                        matrix[i, j] = value
                    return matrix - matrix.T
                start = 0
                for pairs in four_mo_md_pairs(tree):
                    end = start + len(pairs)
                    a, b = generator(first[start:end], pairs), generator(delta[start:end], pairs)
                    assert np.linalg.norm(a @ b - b @ a) > 0, (directory, start, "fixture updates commute")
                    start = end
            elif linear_nodes:
                def generator(values):
                    matrix = np.zeros((3, 3))
                    matrix[0, 1], matrix[0, 2], matrix[1, 2] = values
                    return matrix - matrix.T
                for start in (0, 3):
                    initial_generator, delta_generator = generator(first[start:start + 3]), generator(delta[start:start + 3])
                    assert np.linalg.norm(initial_generator @ delta_generator - delta_generator @ initial_generator) > 0, (
                        directory, start, "fixture updates commute")
            elif args.many_occupied_orbital_ecp:
                def generator(values):
                    matrix = np.zeros((4, 4))
                    matrix[0, 2], matrix[0, 3], matrix[1, 2], matrix[1, 3] = values
                    return matrix - matrix.T
                first_generator, delta_generator = generator(first), generator(delta)
                assert np.linalg.norm(first_generator @ delta_generator - delta_generator @ first_generator) > 0, directory
                # Full coordinates put four OV entries first, then OO and VV.
                assert np.any(orbital_history[-1][1][4:] != 0), (directory, "missing accumulated internal rotations")
            else:
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
                    limits.update(lm_maximum_subspace_dimension="2" if linear_ci else "9")
                    if optimizer == "alm":
                        limits.update(lm_inner_iterations="32", lm_inner_restart_length="2" if linear_ci else "9")
                else:
                    limits.update(gmres_restart_length="16", maximum_damping_retries="0",
                                  pii_energy_undershoot="1.0")
                for name, value in limits.items():
                    ET.SubElement(training, "parameter", name=name).text = value
            try:
                if args.csf_lm and optimizer == "alm":
                    # This protected sample pencil has a complex-conjugate pair.
                    # Refusal is correct; do not tune the fixture to force a root.
                    directory = run(optimizer, optimizer=optimizer, mutate=controls,
                                    samples_per_rank=32, reject="Live LM solve failed: untrusted_root")
                    csv_file = directory / "sj.s000.training_info.csv"
                    assert not csv_file.exists() or not history(directory), directory
                    assert not list(directory.rglob("*.h5")), directory
                    log = (directory / "run.log").read_text()
                    assert "WFTrain refreshed parameter version" not in log, directory
                    assert "WFTrain iteration 0 version" not in log, directory
                    definition = ET.parse(directory / "run.xml")
                    csfs = definition.findall(".//detlist/csf")
                    assert [node.get("coeff") for node in csfs] == ["0.6", "0"]
                    assert all(len(node) == 2 for node in csfs)
                    assert all(node.get("optimize") == "no" for node in definition.findall(
                        "wavefunction/jastrow/correlation/coefficients"))
                    reports.append((optimizer, "expected_untrusted_root", ""))
                    print("alm: expected complex-root refusal before update/publication", flush=True)
                    continue
                directory = run(optimizer, optimizer=optimizer, mutate=controls,
                                samples_per_rank=32, require_committed=ci_workflow)
                if ci_workflow:
                    final = sorted((directory / "sj.s000.training").rglob("*.parameters.h5"))[-1]
                    loaded = run(f"{optimizer}_export_vmc", vmc=True, parameter_file=final)
                    reference = run(f"{optimizer}_explicit_vmc", vmc=True, reference_file=final)
                    np.testing.assert_array_equal(vmc_physical_trace(loaded), vmc_physical_trace(reference))
                if linear_ci or args.csf_pii or mixed_ci:
                    split = run(f"{optimizer}_split", optimizer=optimizer, iterations=1,
                                mutate=controls, samples_per_rank=32)
                    resumed = run(f"{optimizer}_resumed", optimizer=optimizer,
                                  restart=split / "sj.s000.last_checkpoint.h5", mutate=controls, samples_per_rank=32)
                    restored = sorted((resumed / "sj.s000.training").rglob("*.parameters.h5"))[-1]
                    lhs, rhs = datasets(final), datasets(restored)
                    assert lhs.keys() == rhs.keys()
                    for name in lhs:
                        np.testing.assert_array_equal(lhs[name], rhs[name], err_msg=f"pure-CI restart model: {name}")
                    final_bundle = final.parent / (final.name[:-len(".parameters.h5")] + ".checkpoint")
                    resumed_bundle = restored.parent / (restored.name[:-len(".parameters.h5")] + ".checkpoint")
                    for pattern in ("*.state.h5", "*.rank_*.h5"):
                        compare_archives(final_bundle, resumed_bundle, pattern)
                    expected, actual = history(directory)[-1], history(resumed)[0]
                    assert expected.keys() == actual.keys()
                    ignored = {"sampling_seconds", "total_seconds", "resident_bytes"}
                    assert all(expected[key] == actual[key] for key in expected if key not in ignored)
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
        assert all(status == "complete" or (args.csf_lm and method == "alm" and
                   status == "expected_untrusted_root") for method, status, _ in reports), work
        if linear_ci:
            representation = "two-DETS-per-CSF" if args.csf_lm else "pure-DETS"
            methods = "LM positive and ALM pre-update complex-root refusal" if args.csf_lm else "LM/ALM projective updates"
            physics = "ordinary-locality ECP" if md_ecp else "all-electron"
            print(f"PASS: {representation} {physics} {methods}, fixed reference/SPO/J2/expansion, "
                  "exact positive exported inference/restart; AE mixed CI/J2 has separate qualification")
        elif args.ci_pii or args.csf_pii:
            representation = "two-DETS-per-CSF" if args.csf_pii else "pure-DETS"
            print(f"PASS: {representation} PII/minPII committed updates, fixed reference/SPO/J2/expansion and exact exported inference; not mixed CI/Jastrow qualification")
        elif mixed_ci:
            representation = "nontrivial CSF/J2" if args.csf_j2_pii else "DETS/J2"
            physics = "ordinary-locality ECP" if md_ecp else "all-electron"
            print(f"PASS: mixed {representation} {physics} PII/minPII updates, fixed reference/SPO/expansion, "
                  "exact exported inference/restart; mixed LM has a separate bounded workflow")
        else:
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

    adam = run("lm" if args.csf_j2_ecp_lm or args.ci_old_new else "alm" if args.mdj_alm or orbital_alm else "minpii" if args.joint_ci_orbital_ecp_minpii else "adam",
               optimizer="lm" if args.ci_old_new else None, iterations=1 if args.ci_old_new else 2,
               require_committed=not args.ci_old_new,
               samples_per_rank=32 if args.old_new else 4, inference=not args.old_new)
    if args.ci_old_new:
        # Derive the original driver from the SAME fixed scientific definition,
        # retaining its distinct OneShiftOnly shift/rescale policy unchanged.
        old_tree = ET.parse(fixture)
        old_tree.find("project").set("id", "sj")
        old_tree.getroot().remove(old_tree.find("qmc"))
        template = ET.parse(Path(__file__).with_name("linear_sj.xml")).find("loop/qmc")
        old_tree.getroot().append(ET.fromstring(ET.tostring(template)))
        for tag in ("random", "wavefunction", "hamiltonian"):
            assert ET.tostring(old_tree.find(tag)) == ET.tostring(fixture_tree.find(tag)), tag
        assert [ET.tostring(node) for node in old_tree.findall("particleset")] == [
            ET.tostring(node) for node in fixture_tree.findall("particleset")]
        original = work / "original_linear"
        original.mkdir()
        old_tree.write(original / "run.xml")
        result, text = launch(original)
        assert result.returncode == 0, (original, text[-6000:])
        observed = [int(value) for value in re.findall(r"samples per rank\s*=\s*(\d+)", text)]
        assert observed and all(value == 32 for value in observed), (original, observed)
        optimized = sorted(original.glob("sj.s*.opt.xml"))
        assert len(optimized) == 1, (original, optimized)
        old_vp = optimized[0].with_name(optimized[0].name.replace(".opt.xml", ".vp.h5"))
        values = datasets(old_vp)
        names = [text_value(value) for value in values["name_value_lists/parameter_names"].reshape(-1)]
        old_parameters = values["name_value_lists/parameter_values"].reshape(-1)
        assert names == ["CI1"] and old_parameters.shape == (1,) and np.isfinite(old_parameters).all()
        serialized = ET.parse(optimized[0])
        original_ci = serialized.find(".//detlist")
        assert original_ci[0].get("id") == "CI0" and float(original_ci[0].get("coeff")) == 0.6
        assert original_ci[1].get("id") == "CI1"
        np.testing.assert_allclose(float(original_ci[1].get("coeff")), old_parameters[0],
                                   rtol=5e-9, atol=5e-10, err_msg="legacy printed XML precision only")
        for path in ("wavefunction/sposet_collection", "wavefunction/jastrow"):
            before, after = fixture_tree.find(path), serialized.find(path)
            assert after is not None
            if path.endswith("sposet_collection"):
                before, after = before.find("sposet/coefficient"), after.find("sposet/coefficient")
                np.testing.assert_array_equal(np.fromstring(before.text, sep=" "), np.fromstring(after.text, sep=" "))
            else:
                initial_j2, retained_j2 = before.findall("correlation/coefficients"), after.findall("correlation/coefficients")
                assert len(initial_j2) == len(retained_j2) == 2
                for a, b in zip(initial_j2, retained_j2):
                    np.testing.assert_array_equal(np.fromstring(a.text, sep=" "), np.fromstring(b.text, sep=" "))
        scalar = optimized[0].with_name(optimized[0].name.replace(".opt.xml", ".scalar.dat"))
        header = next(line.split()[1:] for line in scalar.read_text().splitlines() if line.startswith("#"))
        old_data = np.loadtxt(scalar, ndmin=2)
        assert old_data.shape[0] == 8 and np.isfinite(old_data).all()
        assert int(old_data[:, header.index("BlockWeight")].sum()) == 32
        model = next((adam / "sj.s000.training").rglob("*.parameters.h5"))
        new_values = datasets(model)
        new_names = [text_value(value) for value in new_values["conventional_wavefunction/variable_names"].reshape(-1)]
        new_parameters = new_values["conventional_wavefunction/parameters"].reshape(-1)
        assert new_names == ["CI1"] and new_parameters.shape == (1,) and np.isfinite(new_parameters).all()
        candidates = dict(old_one_shift=float(old_parameters[0]), new_full_lm=float(new_parameters[0]))
        (work / "candidate_c1.txt").write_text("".join(f"{value:.17g}\n" for value in candidates.values()))
        (work / "candidate_comparison.json").write_text(json.dumps(dict(
            candidates=candidates, frozen_C0=0.6, fixed_J2=True, samples_per_rank=32,
            old_vp=str(old_vp), old_vp_sha256=hashlib.sha256(old_vp.read_bytes()).hexdigest(),
            new_model=str(model), new_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
            old_policy="OneShiftOnly shift_i=1 shift_s=1", new_policy="full LM damping=.01",
            scope="independent native sampling; candidate plumbing only; same-config physical/Rayleigh check pending; no equal/improving update claim"
        ), indent=2) + "\n")
        if args.ci_oracle:
            oracle = work / "candidate_oracle"
            oracle.mkdir()
            candidate_hash = hashlib.sha256((work / "candidate_c1.txt").read_bytes()).hexdigest()
            result, output = launch(oracle, ci_oracle=True)
            assert result.returncode == 0 and "NOT_RUN" not in output, (oracle, output[-6000:])
            matches = re.findall(r"TASK46_NATIVE_CI_CANDIDATES_PASS C1=(\S+),(\S+) physical_Rayleigh=(\S+),(\S+)", output)
            assert len(matches) == 1 and all(np.isfinite(float(value)) for value in matches[0]), oracle
            np.testing.assert_array_equal(np.asarray(matches[0][:2], dtype=float), list(candidates.values()))
            assert hashlib.sha256((work / "candidate_c1.txt").read_bytes()).hexdigest() == candidate_hash
            comparison = json.loads((work / "candidate_comparison.json").read_text())
            comparison.update(oracle_test_sha256=hashlib.sha256(args.ci_oracle.read_bytes()).hexdigest(),
                              candidate_file_sha256=candidate_hash,
                              same_configuration_physical_Rayleigh=list(map(float, matches[0][2:])),
                              scope="independently sampled native candidates; same-state physical/dense objective checked; no quality ranking")
            (work / "candidate_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
            print("PASS actual CI candidates: native receipts plus same-configuration physical/dense Rayleigh oracle")
            return
        print("PASS candidate plumbing only: actual pure-CI OneShiftOnly/LM scalars mapped by CI1; physical oracle pending")
        return
    if args.periodic_heg or args.md_orbital:
        def sr_controls(tree):
            limits = {"damping": "0.01", "krylov_maximum_iterations": "32",
                      "krylov_maximum_operator_applications": "128", "krylov_maximum_seconds": "10",
                      "krylov_relative_tolerance": "1e-8", "maximum_damping_attempts": "1"}
            for name, value in limits.items():
                ET.SubElement(tree.find("qmc/training"), "parameter", name=name).text = value
        run("sr", optimizer="sr", mutate=sr_controls)
    if args.old_new:
        old_fixture = fixture.with_name("linear_sj.xml")
        old_tree = ET.parse(old_fixture)
        new_tree = ET.parse(fixture)
        for tag in ("random", "wavefunction", "hamiltonian"):
            assert ET.tostring(old_tree.find(tag)) == ET.tostring(new_tree.find(tag)), tag
        assert [ET.tostring(node) for node in old_tree.findall("particleset")] == [
            ET.tostring(node) for node in new_tree.findall("particleset")]
        original = work / "original_linear"
        original.mkdir()
        old_tree.find("project").set("id", "sj")
        old_tree.write(original / "run.xml")
        result, text = launch(original)
        assert result.returncode == 0, (original, text[-6000:])
        assert "Building overlap and Hamiltonian matrices" in text and "Solving eigenvalue problem" in text
        observed = [int(value) for value in re.findall(r"samples per rank\s*=\s*(\d+)", text)]
        assert observed and all(value == 32 for value in observed), (original, observed)
        optimized = sorted(original.glob("sj.s*.opt.xml"))
        assert len(optimized) == 2, (original, optimized)
        initial = {f"{node.get('id')}_{i}": float(value)
                   for node in old_tree.findall("wavefunction/jastrow/correlation/coefficients")
                   for i, value in enumerate(node.text.split())}
        assert len(initial) == 8
        previous = initial
        old_samples = []
        for xml in optimized:
            vp = xml.with_name(xml.name.replace(".opt.xml", ".vp.h5"))
            values = datasets(vp)
            names = [text_value(name) for name in values["name_value_lists/parameter_names"].reshape(-1)]
            active = dict(zip(names, values["name_value_lists/parameter_values"].reshape(-1)))
            assert active.keys() == initial.keys() and all(math.isfinite(value) for value in active.values()), vp
            assert any(active[name] != previous[name] for name in active), vp
            previous = active
            serialized = ET.parse(xml)
            for node in serialized.findall("wavefunction/jastrow/correlation/coefficients"):
                np.testing.assert_allclose([float(value) for value in node.text.split()],
                    [active[f"{node.get('id')}_{i}"] for i in range(len(node.text.split()))],
                    rtol=5e-9, atol=5e-10, err_msg="legacy printed XML precision")
            fixed = old_tree.find(".//sposet/coefficient")
            retained = serialized.find(".//sposet/coefficient")
            assert retained is not None and retained.attrib == fixed.attrib
            np.testing.assert_array_equal([float(value) for value in retained.text.split()],
                                          [float(value) for value in fixed.text.split()])
            scalar = xml.with_name(xml.name.replace(".opt.xml", ".scalar.dat"))
            header = next(line.split()[1:] for line in scalar.read_text().splitlines() if line.startswith("#"))
            data = np.loadtxt(scalar, ndmin=2)
            assert data.shape[0] == 8 and np.isfinite(data).all(), scalar
            old_samples.append(int(data[:, header.index("BlockWeight")].sum()))
        assert old_samples == [32, 32], old_samples
        old_final = optimized[-1].with_name(optimized[-1].name.replace(".opt.xml", ".vp.h5"))
        old_loaded = run("old_export_vmc", vmc=True, legacy_vp=old_final)
        old_reference = run("old_explicit_vmc", vmc=True, legacy_reference=old_final)
        np.testing.assert_array_equal(vmc_physical_trace(old_loaded), vmc_physical_trace(old_reference))
        new_final = sorted((adam / "sj.s000.training").rglob("*.parameters.h5"))[-1]
        new_loaded = run("new_export_vmc", vmc=True, parameter_file=new_final)
        new_reference = run("new_explicit_vmc", vmc=True, reference_file=new_final)
        np.testing.assert_array_equal(vmc_physical_trace(new_loaded), vmc_physical_trace(new_reference))
        digest = hashlib.sha256()
        with executable.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        (work / "comparison.json").write_text(json.dumps({
            "executable_sha256": digest.hexdigest(), "original": "batched linear OneShiftOnly",
            "new": "WFTrain Adam", "active_parameters": 8, "updates_each": 2,
            "old_observed_samples_each": old_samples, "new_observed_samples_each": [int(row["samples"]) for row in history(adam)],
            "initial_ansatz_equal": True, "reusable_exports_equal_to_explicit_physical_reference": True,
            "scope": "finite changed states and reusable inference; no equal updates/convergence/equal-quality timing claim"
        }, indent=2) + "\n")
        print("PASS: original OneShiftOnly/new Adam finite changed states, observed samples and reusable native VMC exports")
        return
    if args.model_import:
        exported = sorted((adam / "sj.s000.training").rglob("*.parameters.h5"))[-1]
        imported_vmc = run("imported_vmc", vmc=True, parameter_file=exported)
        explicit_vmc = run("explicit_vmc", vmc=True, reference_file=exported)
        compare_vmc_reference(imported_vmc, explicit_vmc)
        trained = run("imported_training", parameter_file=exported)
        split = run("imported_split", iterations=1, parameter_file=exported)
        resumed = run("imported_resumed", parameter_file=exported,
                      restart=split / "sj.s000.last_checkpoint.h5")
        for name in ("samples", "mean_energy", "variance", "gradient_norm", "selected_alpha", "committed"):
            assert history(trained)[-1][name] == history(resumed)[0][name], name
        if args.md_orbital or args.mdj_alm or orbital_alm or args.sj_reference_line_search or args.periodic_spline:
            original_row, resumed_row = history(trained)[-1], history(resumed)[0]
            diagnostic_fields = {"sampling_seconds", "total_seconds", "retained_numeric_bytes", "resident_bytes"}
            assert original_row.keys() == resumed_row.keys()
            for name in original_row.keys() - diagnostic_fields:
                assert original_row[name] == resumed_row[name], (name, original_row[name], resumed_row[name])
        final = sorted((trained / "sj.s000.training").rglob("*.parameters.h5"))[-1]
        restored = sorted((resumed / "sj.s000.training").rglob("*.parameters.h5"))[-1]
        lhs, rhs = datasets(final), datasets(restored)
        assert lhs.keys() == rhs.keys()
        for name in lhs:
            np.testing.assert_array_equal(lhs[name], rhs[name], err_msg=f"initialized restart: {name}")
        final_bundle = final.parent / (final.name[:-len(".parameters.h5")] + ".checkpoint")
        restored_bundle = restored.parent / (restored.name[:-len(".parameters.h5")] + ".checkpoint")
        if args.md_orbital:
            # Validate each serialized optimizer state with production restore,
            # including its OWN fingerprint. Do not sample or apply an update.
            for label, directory in (("validate_full_optimizer", trained), ("validate_resumed_optimizer", resumed)):
                run(label, parameter_file=exported, iterations=2, completed_restore=True,
                    restart=directory / "sj.s000.last_checkpoint.h5")
        for pattern in ("*.state.h5", "*.rank_*.h5"):
            compare_archives(final_bundle, restored_bundle, pattern,
                             optimizer_max_ulps=8 if args.md_orbital and pattern == "*.state.h5" else 0)
        if orbital_alm:
            for bundle in (final_bundle, restored_bundle):
                shards = sorted(bundle.glob("*.rank_*.h5"))
                rank_ids = [int(re.search(r"\.rank_(\d+)\.h5$", shard.name).group(1)) for shard in shards]
                assert rank_ids == list(range(args.ranks)), (bundle, rank_ids)
            assert all(int(row["samples"]) == 32 * args.ranks
                       for directory in (adam, trained, split, resumed) for row in history(directory))
        # A second import still uses ORIGINAL inline XML and original registration
        # identity, rather than a rewritten XML whose baseline masks a loader bug.
        second_vmc = run("reimported_vmc", vmc=True, parameter_file=final)
        second_reference = run("reexport_reference_vmc", vmc=True, reference_file=final)
        compare_vmc_reference(second_vmc, second_reference)
        for name in ("model_identity", "registration_values", "wavefunction_xml", "physical_definition"):
            np.testing.assert_array_equal(datasets(exported)["conventional_wavefunction/" + name],
                                          lhs["conventional_wavefunction/" + name])
        missing = work / "missing-parameters.h5"
        shutil.copyfile(exported, missing)
        with h5py.File(missing, "r+") as handle:
            del handle["conventional_wavefunction/parameters"]
        run("reject_missing_import_parameters", parameter_file=missing, vmc=True,
            reject="Conventional artifact has an invalid or oversized required field: parameters")
        run("reject_unsupported_import_driver", parameter_file=exported, vmc=True,
            mutate=lambda tree: tree.find("qmc").set("method", "dmc"),
            reject="parameter_file supports only CPU batched VMC and WFTrain")
        if args.md_orbital:
            def one_active_spin(tree):
                collection = tree.find("wavefunction/sposet_collection")
                wrapper = collection.findall("rotated_sposet")[1]
                spo = wrapper.find("sposet")
                spo.set("name", wrapper.get("name"))
                collection.remove(wrapper)
                collection.append(spo)
            run("reject_one_active_md_spin", mutate=one_active_spin,
                reject="Optimizing the SPOSet of only only species is not supported!")
        if periodic_profile:
            for label, name, value in (("cell", "rs", "5.1"), ("cutoff", "LR_dim_cutoff", "7")):
                def change_periodic_identity(tree, name=name, value=value):
                    tree.find(f".//simulationcell/parameter[@name='{name}']").text = value
                run(f"reject_changed_{label}_import", parameter_file=exported, vmc=True,
                    mutate=change_periodic_identity,
                    reject=("Conventional model export identity mismatch",
                            "Conventional artifact has an invalid or oversized required field: model_identity")
                           if args.complex_periodic_heg else "Conventional model export identity mismatch")
                run(f"reject_changed_{label}_restart", restart=adam / "sj.s000.last_checkpoint.h5", iterations=3,
                    mutate=change_periodic_identity,
                    reject="Training checkpoint stage identity or configuration mismatch")
        if args.external_lcao:
            def route_lcao(path):
                return lambda tree: tree.find("wavefunction/sposet_collection").set("href", str(path))
            relocated = external_source if args.external_orbital_ecp else work / "relocated-lcao.h5"
            if not args.external_orbital_ecp:
                shutil.copyfile(external_asset, relocated)
            relocated_vmc = run("relocated_import_vmc", vmc=True, parameter_file=exported,
                                mutate=route_lcao(relocated))
            np.testing.assert_array_equal(vmc_physical_trace(imported_vmc), vmc_physical_trace(relocated_vmc))
            # Mutate copies only: both the loaded basis and loaded coefficient
            # values must participate in physical identity, not just a filename.
            for label, dataset in (("basis", "basisset/atomicBasisSet0/basisGroup0/radfunctions/DataRad0/exponent"),
                                   ("coefficients", "Super_Twist/eigenset_0")):
                changed = external_asset if args.external_orbital_ecp else work / f"changed-{label}.h5"
                if not args.external_orbital_ecp:
                    shutil.copyfile(external_asset, changed)
                with h5py.File(changed, "r+") as handle:
                    if label == "basis":
                        candidates = []
                        handle["basisset"].visititems(lambda name, node: candidates.append(node.name)
                            if isinstance(node, h5py.Dataset) and name.endswith("exponent") else None)
                        assert candidates, changed
                        dataset = candidates[0]
                    node = handle[dataset]
                    value = np.asarray(node[()])
                    value.flat[0] += 0.001
                    node[...] = value
                try:
                    if args.external_orbital_ecp:
                        (work / f"changed-{label}.sha256").write_text(hashlib.sha256(changed.read_bytes()).hexdigest() + "\n")
                    run(f"reject_changed_{label}_import", vmc=True, parameter_file=exported,
                        mutate=route_lcao(changed), reject="Conventional model export identity mismatch")
                    run(f"reject_changed_{label}_restart", iterations=3, restart=adam / "sj.s000.last_checkpoint.h5",
                        mutate=route_lcao(changed),
                        reject=("Conventional orbital frame shape or reference identity mismatch"
                                if args.external_orbital_ecp and label == "coefficients" else
                                "Training checkpoint stage identity or configuration mismatch"))
                finally:
                    if args.external_orbital_ecp:
                        external_asset.write_bytes(external_original)
            def remove_capture(tree):
                collection = tree.find("wavefunction/sposet_collection")
                del collection.attrib["source_identity"]
                del collection.attrib["source_identity_max_bytes"]
            run("reject_uncaptured_lcao", mutate=remove_capture, reject="external")
            assert external_source.read_bytes() == external_original
            assert external_asset.read_bytes() == external_original
            print(f"PASS: external CPU LCAO{' active P6/full-history VP physical reference tol=3e-12 ONLY; bitwise accepted-C restart;' if args.external_orbital_ecp else ''}"
                  f"{' plus ordinary-locality ECP' if external_ecp else ''} Adam/model reuse, exact restart and relocation, basis/coefficient mutation rejection; "
                  f"immutable asset bytes={len(external_original)} (separate from derivative scratch)")
        if args.periodic_spline:
            def tighten_spline_cutoff(tree):
                tree.find(".//simulationcell/parameter[@name='LR_dim_cutoff']").text = "32"
            run("reject_changed_spline_cutoff_import", vmc=True, parameter_file=exported, mutate=tighten_spline_cutoff,
                reject=("Conventional model export identity mismatch",
                        "Conventional artifact has an invalid or oversized required field: model_identity",
                        "Conventional artifact has an invalid or oversized required field: physical_definition"))
            run("reject_changed_spline_cutoff_restart", iterations=3, restart=adam / "sj.s000.last_checkpoint.h5",
                mutate=tighten_spline_cutoff, reject="Training checkpoint stage identity or configuration mismatch")
        pseudos = ET.parse(fixture).findall("hamiltonian/pairpot/pseudo")
        for pseudo in pseudos:
            species = pseudo.get("elementType")
            source = (fixture.parent / pseudo.get("href")).resolve()
            original = source.read_bytes()
            changed = work / f"changed-import-pseudopotential-{species}.xml"
            changed.write_bytes(original + b"\n<!-- imported model PP identity regression -->\n")
            run("reject_changed_import_pp" + (f"_{species}" if len(pseudos) > 1 else ""), parameter_file=exported, vmc=True,
                mutate=lambda tree: tree.find(f"hamiltonian/pairpot/pseudo[@elementType='{species}']").set("href", str(changed)),
                # A changed variable-width hex fingerprint may fail the trusted
                # model_identity string-width guard before content comparison.
                reject=("Conventional model export identity mismatch",
                        "Conventional artifact has an invalid or oversized required field: model_identity"))
            assert source.read_bytes() == original
        if args.periodic_spline:
            assert spline_source.read_bytes() == spline_image
            print("PASS: real Gamma diamond fixed-spline/J1/J2 P24 Adam, actual source/used-table and Ewald30, "
                  "ordinary rule4 locality ECP, physical model reuse and strict exact initialized restart")
        elif args.complex_periodic_heg:
            print("PASS: CPU complex Gamma HEG14 real-J2 artifact VMC reference, fresh training/re-export/re-import, "
                  "complete exact initialized split restart and changed identity/import rejection")
        elif args.periodic_heg:
            print("PASS: real Gamma HEG14 Adam/SR, 8 samples/update, actual Ewald identity and changed cell/cutoff rejection")
        if args.md_orbital:
            print("PASS: distinct-spin MD orbital/CI/J2 P11 Adam/SR, noncommuting full frames and early one-active-spin rejection; "
                  "physical/model/sampler/scientific CSV exact restart, own optimizer payloads validated, "
                  "legacy cold-process moments <=8 ULP (NOT bitwise optimizer restart)")
        elif args.sj_reference_line_search:
            mode = "endpoint" if args.sj_endpoints_line_search else "reference"
            detail = "finite candidate receipts" if args.sj_endpoints_line_search else "five nonzero alphas, finite candidate receipts"
            print(f"PASS: H2 SJ Adam {mode} search, {detail}, "
                  "natural decisions/accepted-state preservation and strict exact initialized restart")
        elif args.mdj_alm:
            profile = ("Be signed-CSF/J2 locality-ECP LM P5, outer CSF0=.6/expanded C0=.48"
                       if args.csf_j2_ecp_lm else "Be DETS/J2 locality-ECP ALM P5"
                       if args.mdj_ecp_alm else "H4 DETS/J2 AE ALM P9")
            print(f"PASS: real CPU fixed-orbital {profile}, two committed {32 * args.ranks}-sample updates, "
                  "frozen C0/SPO, exported inference/model reuse and strict exact initialized restart")
        elif args.csf_orbital_ecp:
            if args.csf_orbital_ecp_alm:
                run("reject_fixed_csf_orbital_alm", optimizer="alm",
                    mutate=lambda tree: tree.find(".//multideterminant").set("optimize", "no"),
                    reject="mixed updates remain unqualified")
                run("reject_csf_orbital_minpii", optimizer="minpii", reject="mixed updates remain unqualified")
            method = "ALM" if args.csf_orbital_ecp_alm else "Adam"
            print(f"PASS: real CPU signed-child CSF/orbital/J2 ECP {method} P17, frozen outer CSF0=.6/expanded C0=.48, "
                  "immutable child gauge, both frames, physical VMC reuse and strict exact initialized restart")
        elif args.unequal_spin_orbital_ecp:
            print("PASS: real CPU charged C^- 3up/2down joint CI1/orbital/J2 ECP P16, "
                  "five/six-pair frames, frozen C0/original XML and strict exact initialized restart")
        elif args.joint_ci_orbital_ecp:
            if args.joint_ci_orbital_ecp_alm:
                run("reject_fixed_ci_orbital_alm", optimizer="alm",
                    mutate=lambda tree: tree.find(".//multideterminant").set("optimize", "no"),
                    reject="mixed updates remain unqualified")
            method = "ALM" if args.joint_ci_orbital_ecp_alm else "minPII" if args.joint_ci_orbital_ecp_minpii else "Adam"
            print(f"PASS: real CPU joint CI1/distinct-spin orbital/J2 ordinary-locality ECP {method} P17, "
                  "frozen C0/original XML, two complete frames, physical VMC reuse and strict exact initialized restart")
            if args.joint_ci_orbital_ecp_alm:
                print(f"PASS: P17 ALM {args.ranks} MPI x 1 OpenMP, {32 * args.ranks} global samples/update, "
                      f"{args.ranks} exact initialized-restart sampler shards", flush=True)
        elif args.md_orbital_ecp:
            print("PASS: real CPU fixed-CI distinct-spin MD orbital/J2 ordinary-locality ECP Adam P16, "
                  "two complete 4AO/4MO frames, physical VMC model reuse and strict exact initialized restart")
        elif args.many_occupied_orbital_ecp:
            print("PASS: real CPU neutral-C two-occupied orbital/J2 ordinary-locality ECP Adam P8, 4AO/4MO full frames, "
                  "physical VMC model reuse and exact initialized restart; numerical coverage is the separate unit gate")
        elif args.orbital_ecp:
            print("PASS: real CPU one-occupied Be orbital/J2 ordinary-locality ECP Adam, 4AO/3MO full frames, "
                  "physical VMC model reuse and exact initialized restart; mathematical nodal coverage is the separate unit gate")
        else:
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
    if not args.complex_periodic_heg and not csf_ecp and not j3_ecp:
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
        if md_ecp or j3_ecp or args.complex_periodic_heg:
            final_bundle = final.parent / (final.name[:-len(".parameters.h5")] + ".checkpoint")
            restored_bundle = restored.parent / (restored.name[:-len(".parameters.h5")] + ".checkpoint")
            for pattern in ("*.state.h5", "*.rank_*.h5"):
                compare_archives(final_bundle, restored_bundle, pattern)
            expected, actual = history(adam)[-1], history(resumed)[0]
            assert expected.keys() == actual.keys()
            ignored = {"sampling_seconds", "total_seconds", "resident_bytes"}
            assert all(expected[key] == actual[key] for key in expected if key not in ignored)
    if args.preflight:
        run("reject_kfac", optimizer="kfac", reject="no affine observations required by KFAC")
        run("reject_zero_cap", reject="legacy_maximum_parameters must be positive", mutate=lambda tree: setattr(
            tree.find("qmc/training/parameter[@name='legacy_maximum_parameters']"), "text", "0"))

        def neural_decay(tree):
            ET.SubElement(tree.find("qmc/training"), "parameter", name="weight_decay").text = "0.01"
        run("reject_neural_decay", reject="weight_decay must be zero", mutate=neural_decay)
        if args.complex_periodic_heg:
            reason = "Conventional complex training requires Gamma HEG14 ENERGY Adam"
            run("reject_complex_sgd", optimizer="sgd", reject=reason)
            run("reject_complex_line_search", reject=reason, mutate=lambda tree: setattr(
                tree.find("qmc/training/parameter[@name='line_search']"), "text", "reference"))
            run("reject_complex_twist", reject="does not admit twist", mutate=lambda tree: tree.find(
                "wavefunction/sposet_collection/sposet").set("twist", "0.1 0 0"))
            mismatch = "Training checkpoint stage identity or configuration mismatch"
            for label, parameter, value in (("cell", "rs", "5.1"), ("cutoff", "LR_dim_cutoff", "7")):
                run(f"reject_complex_changed_{label}", iterations=3,
                    restart=adam / "sj.s000.last_checkpoint.h5", reject=mismatch,
                    mutate=lambda tree, key=parameter, number=value: setattr(
                        tree.find(f"qmcsystem/simulationcell/parameter[@name='{key}']"), "text", number))
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
    methods = ("CPU complex Gamma HEG14 Adam" if args.complex_periodic_heg else
               "CPU CSF/J2 ordinary-locality ECP Adam" if csf_ecp else
               "CPU fixed-Slater J2/J3 ordinary-locality ECP Adam" if j3_ecp else f"Adam/{args.companion_optimizer}")
    completed = f"{methods} conventional updates, interval inference, immutable model reference, exact inference isolation"
    if args.restart:
        completed += ", exact split restart"
    if args.preflight:
        completed += ", expected preflight diagnostics"
    print("PASS: " + completed)


if __name__ == "__main__":
    main()
