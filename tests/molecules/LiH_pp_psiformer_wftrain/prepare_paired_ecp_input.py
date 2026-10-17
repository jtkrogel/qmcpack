#!/usr/bin/env python3
"""Prepare, never execute, the proposed internally consistent periodic LiH BFD pilot."""

import argparse, copy, hashlib, json
from pathlib import Path
import xml.etree.ElementTree as ET

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
real_input = (
    a.source / "tests/molecules/LiH_pp_psiformer_wftrain/wftrain_ecp_checkpoint.xml"
)
periodic_input = (
    a.source / "tests/molecules/H2_ae_psiformer_wftrain/wftrain_complex_orbital_mse.xml"
)
tree = ET.parse(real_input)
root = tree.getroot()
original = copy.deepcopy(root)
root.find("project").set("id", "paired_lih_bfd_energy")
system = ET.Element("qmcsystem")
system.append(copy.deepcopy(ET.parse(periodic_input).find("qmcsystem/simulationcell")))
cell = system.find("simulationcell")
# Long-range electrostatic handler/cutoff, not the optimizer learning rate.
for name, value in (("LR_handler", "ewald"), ("LR_dim_cutoff", "30")):
    node = cell.find(f"parameter[@name='{name}']")
    if node is None:
        node = ET.SubElement(cell, "parameter", name=name)
    node.text = value
for node in list(root):
    if node.tag in ("particleset", "wavefunction", "hamiltonian"):
        root.remove(node)
        system.append(node)
root.insert(2, system)  # QMCMain reads the cell through qmcsystem, before all P/W/H.
head = system.find("wavefunction/psiformer")
for key in ("optimize_indices", "optimize_scope"):
    head.attrib.pop(key, None)
head.attrib.update(
    initialization="psiformer_complex_periodic_v1",
    initialization_seed="37",
    feature_policy="periodic_torus_v1",
    system="pseudopotential",
    gpu="cuda",
    optimize="no",
    twist="0 0 0",
    max_clone_numeric_bytes=str(16 * 1024**2),
    max_gpu_numeric_bytes=str(512 * 1024**2),
)
for op in system.findall("hamiltonian/pairpot"):
    if op.get("type") == "coulomb":
        op.set("gpu", "no")
qmc = root.find("qmc")
qmc.set("move", "pbyp")
training = qmc.find("training")
training.attrib.update(objective="energy", optimizer="adam")


def parameter(parent, name, value):
    node = parent.find(f"parameter[@name='{name}']")
    if node is None:
        node = ET.SubElement(parent, "parameter", name=name)
    node.text = str(value)


for key, value in dict(
    iterations=2,
    line_search="off",
    maximum_retained_bytes=1024**3,
    complex_gpu_device_bytes=512 * 1024**2,
    complex_gpu_host_bytes=512 * 1024**2,
    complex_gpu_kinetic_targets=1,
    complex_gpu_maximum_matrix_work=10**9,
).items():
    parameter(training, key, value)
for key, value in dict(
    walkers_per_rank=1, crowds=1, blocks=1, steps=1, warmupsteps=0
).items():
    parameter(qmc, key, value)
assert root.find("random").get("seed") == "91427"
assert training.find("parameter[@name='learning_rate']").text.strip() == "1e-10"
assert system.find("particleset[@name='e']/group[@name='u']").get("size") == "1"
assert system.find("particleset[@name='e']/group[@name='d']").get("size") == "1"
for element, z in (("Li", 3), ("H", 1)):
    group = system.find(f"particleset[@name='ion0']/group[@name='{element}']")
    assert float(group.find("parameter[@name='charge']").text) == 1
    assert float(group.find("parameter[@name='valence']").text) == 1
    assert float(group.find("parameter[@name='atomicnumber']").text) == z
assert system.find("hamiltonian/pairpot[@type='pseudo']").get("DLA") == "no"
assert cell.find("parameter[@name='LR_handler']").text == "ewald"
assert cell.find("parameter[@name='LR_dim_cutoff']").text == "30"
assert (
    len(system.findall("hamiltonian/pairpot[@type='coulomb']")) == 2
)  # EE+II, never AE electron-ion.
a.output.mkdir(parents=True, exist_ok=False)
for element in ("Li", "H"):
    (a.output / f"{element}.BFD.xml").symlink_to(
        (a.source / f"tests/molecules/LiH_pp/{element}.BFD.xml").resolve()
    )
tree.write(a.output / "input.xml", encoding="utf-8", xml_declaration=True)
manifest = {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (
        real_input,
        periodic_input,
        *(a.source / f"tests/molecules/LiH_pp/{e}.BFD.xml" for e in ("Li", "H")),
    )
}
(a.output / "preparation.json").write_text(
    json.dumps(
        dict(
            status="PREPARED_NOT_RUN",
            inputs=manifest,
            scope="Gamma periodic LiH BFD, neutral valence1+1/Ne2; source metadata adaptation, not AE plus PP.",
        ),
        indent=2,
    )
    + "\n"
)
print("PREPARED_NOT_RUN", a.output)
