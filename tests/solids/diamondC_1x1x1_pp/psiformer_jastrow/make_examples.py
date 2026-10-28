#!/usr/bin/env python3
"""Generate complete CPU Gamma diamond product workflows from packaged reference assets."""
import argparse
import copy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET


def parameter(node, name, value):
    ET.SubElement(node, 'parameter', name=name).text = str(value)


def make_document(asset_dir, project, frozen):
    reference = ET.parse(asset_dir / 'det_qmc.in.xml').getroot()
    root = ET.Element('simulation')
    project_node = ET.SubElement(root, 'project', id=project, series='0')
    parameter(project_node, 'driver_version', 'batched')
    ET.SubElement(root, 'random', seed='73')
    system = ET.SubElement(root, 'qmcsystem')
    source = reference.find('qmcsystem')
    cell = copy.deepcopy(source.find('simulationcell'))
    cell.find("parameter[@name='LR_dim_cutoff']").text = '30'
    system.append(cell)
    for particles in source.findall('particleset'):
        system.append(copy.deepcopy(particles))
    teacher = copy.deepcopy(source.find('wavefunction'))
    teacher.set('name', 'sj_reference')
    teacher.set('print_parameters', 'no')
    teacher.find('sposet_collection').set('href', str(asset_dir / 'pwscf.pwscf.h5'))
    for coefficients in teacher.findall('.//coefficients'):
        coefficients.set('optimize', 'no')
    student = ET.SubElement(system, 'wavefunction', name='student', target='e', print_parameters='no')
    ET.SubElement(student, 'psiformer', name='neural', initialization='psiformer_periodic_jastrow_v1',
                  initialization_seed='37', feature_policy='periodic_torus_v1', source='ion0', system='pseudopotential', gpu='no',
                  neural_jastrow='zero_residual_pair_v1', neural_jastrow_cutoff='1.5', optimize='no')
    for jastrow in source.find('wavefunction').findall('jastrow'):
        term = copy.deepcopy(jastrow)
        for coefficients in term.findall('.//coefficients'):
            coefficients.set('optimize', 'no' if frozen else 'yes')
        student.append(term)
    system.append(teacher)
    hamiltonian = ET.SubElement(system, 'hamiltonian', name='h0', type='generic', target='e')
    ET.SubElement(hamiltonian, 'pairpot', name='ElecElec', type='coulomb', source='e', target='e')
    ET.SubElement(hamiltonian, 'pairpot', name='IonIon', type='coulomb', source='ion0', target='ion0')
    ecp = ET.SubElement(hamiltonian, 'pairpot', name='PseudoPot', type='pseudo', source='ion0',
                        wavefunction='student', format='xml', algorithm='batched', DLA='no')
    ET.SubElement(ecp, 'pseudo', elementType='C', href=str(asset_dir / 'C.BFD.xml'))
    return root


def sampling(node, acceptance, inference=False):
    for name, value in dict(walkers_per_rank=4 if acceptance else 16, crowds=1,
                            blocks=1 if acceptance else 2, steps=2 if acceptance else 8,
                            warmupsteps=4 if acceptance else 64, substeps=1 if acceptance else 4,
                            timestep=.02, usedrift='no').items():
        parameter(node, name, value)


def training(root, objective, optimizer, acceptance):
    qmc = ET.SubElement(root, 'qmc', method='wf_train', move='pbyp', checkpoint='-1')
    ET.SubElement(qmc, 'qmcsystem', wavefunction='student', hamiltonian='h0')
    controls = ET.SubElement(qmc, 'training', objective=objective, optimizer=optimizer)
    values = dict(iterations=3 if acceptance else 10, learning_rate=1e-6 if acceptance else 1e-5,
                  checkpoint_interval=1, parameter_chunk_size=4096, maximum_retained_bytes=2147483648,
                  line_search='off')
    if objective == 'orbital_mse':
        values.update(teacher='sj_reference', orbital_sampling='student', orbital_target='single_reference',
                      target_provenance='packaged-diamond8e-bfd-gamma-spline-v1', learning_rate=1e-4)
    else:
        values.update(gradient_estimator='score_covariance')
        if optimizer == 'kfac':
            values.update(kfac_persist_state='yes', kfac_curvature='native_score', damping=.01,
                          kfac_fallback_damping=.01, maximum_update_norm=.05)
    for name, value in values.items():
        parameter(controls, name, value)
    sampling(qmc, acceptance)
    inference = ET.SubElement(qmc, 'inference', interval='1', last='yes', move='pbyp', evaluate_energy='yes')
    sampling(inference, acceptance, inference=True)
    return qmc


def write(root, path):
    ET.indent(root, space='  ')
    encoded = ET.tostring(root, encoding='unicode', xml_declaration=True)
    path.write_text('\n'.join(line.rstrip() for line in encoded.splitlines())+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--acceptance', action='store_true', help='Use the bounded few-update functional cases.')
    args = parser.parse_args()
    assets = args.asset_dir.resolve()
    for name in ('det_qmc.in.xml', 'pwscf.pwscf.h5', 'C.BFD.xml'):
        if not (assets / name).is_file():
            parser.error(f'Missing packaged asset: {assets / name}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = []
    for name in ('det_qmc.in.xml', 'pwscf.pwscf.h5', 'C.BFD.xml'):
        hashes.append(f'{hashlib.sha256((assets/name).read_bytes()).hexdigest()}  {assets/name}')
    (args.output_dir / 'reference_assets.sha256').write_text('\n'.join(hashes)+'\n')
    root = make_document(assets, 'orbital_pretrain', frozen=False)
    training(root, 'orbital_mse', 'adam', args.acceptance)
    write(root, args.output_dir / 'orbital_pretrain.xml')
    for optimizer in ('adam', 'kfac'):
        for frozen in (True, False):
            name = f'energy_{optimizer}_j_{"frozen" if frozen else "active"}'
            root = make_document(assets, name, frozen)
            training(root, 'energy', optimizer, args.acceptance)
            write(root, args.output_dir / f'{name}.xml')
    for reference in (False, True):
        name = 'sj_reference_vmc' if reference else 'initial_product_vmc'
        root = make_document(assets, name, frozen=True)
        if reference:
            pseudo = root.find(".//pairpot[@type='pseudo']")
            pseudo.set('wavefunction', 'sj_reference')
        qmc = ET.SubElement(root, 'qmc', method='vmc', move='pbyp', checkpoint='-1')
        ET.SubElement(qmc, 'qmcsystem', wavefunction='sj_reference' if reference else 'student', hamiltonian='h0')
        sampling(qmc, args.acceptance)
        write(root, args.output_dir / f'{name}.xml')


if __name__ == '__main__':
    main()
