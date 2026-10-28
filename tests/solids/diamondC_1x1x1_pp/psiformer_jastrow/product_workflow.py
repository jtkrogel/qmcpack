#!/usr/bin/env python3
"""Compose native training stages or extract a matching whole-product model input."""
import argparse
import copy
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def write(root, path):
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def handoff(args):
    pretrain = ET.parse(args.pretrain).getroot()
    energy = ET.parse(args.energy).getroot()
    source = pretrain.findall('qmc')
    target = energy.findall('qmc')
    if len(source) != 1 or len(target) != 1:
        raise ValueError('Each stage must contain exactly one QMC block.')
    if source[0].find('training').get('objective') != 'orbital_mse' or target[0].find('training').get('objective') != 'energy':
        raise ValueError('Expected orbital_mse followed by energy.')
    # The energy stage supplies the physical model and its conventional optimize flags.
    # One process carries the fitted model, current walkers and RNGs between drivers.
    energy.remove(target[0])
    energy.append(copy.deepcopy(source[0]))
    energy.append(target[0])
    energy.find('project').set('id', 'pretrain_to_'+Path(args.energy).stem)
    write(energy, args.output)
    print(f'Wrote {args.output}: one native process, fresh optimizer history for each objective.')
    if args.run:
        subprocess.run([str(args.run.resolve()), str(args.output.resolve())], check=True)


def extract(args):
    import h5py
    def scalar_text(dataset):
        value = dataset[...].reshape(-1)[0]
        return value.decode() if isinstance(value, bytes) else str(value)
    root = ET.parse(args.template).getroot()
    with h5py.File(args.model, 'r') as model:
        if 'jastrow_neural_product' in model:
            if args.pin_template_jastrows:
                raise ValueError('A whole-product association is present; preserve its saved Jastrows.')
            product = model['jastrow_neural_product']
            if scalar_text(product['completion_cookie']) != 'jastrow-neural-product-complete-v1':
                raise ValueError('Incomplete whole-product model.')
            current = ET.fromstring(scalar_text(product['complete_wavefunction_xml']))
        else:
            if not args.pin_template_jastrows:
                raise ValueError('Neural-only model: explicitly use --pin-template-jastrows for a new physical association.')
            if scalar_text(model['psiformer_model/profile']) != 'psiformer_periodic_jastrow_v1':
                raise ValueError('Neural-only association requires the residual profile; convert a legacy model explicitly first.')
            candidates = [node for node in root.findall('.//wavefunction') if node.find('psiformer') is not None]
            if len(candidates) != 1:
                raise ValueError('Expected exactly one neural product in the template.')
            current = copy.deepcopy(candidates[0])
            for node in current.findall('psiformer'):
                for attribute in ('initialization', 'initialization_seed', 'neural_jastrow', 'neural_jastrow_cutoff',
                                  'neural_jastrow_same_reference', 'neural_jastrow_opposite_reference', 'export_parameters'):
                    node.attrib.pop(attribute, None)
    neural = current.findall('psiformer')
    if len(neural) != 1:
        raise ValueError('Product association must contain exactly one native neural factor.')
    neural[0].set('parameters', str(args.model.resolve()))
    candidates = [node for node in root.findall('.//wavefunction') if node.get('name') == current.get('name')]
    if len(candidates) != 1:
        raise ValueError('Template must contain exactly one matching product wavefunction.')
    for parent in root.iter():
        if candidates[0] in list(parent):
            index = list(parent).index(candidates[0])
            parent.remove(candidates[0])
            parent.insert(index, current)
            break
    if args.asset_dir:
        for tag in ('sposet_collection', 'pseudo'):
            for node in root.findall('.//'+tag):
                if node.get('href'):
                    asset = args.asset_dir.resolve()/Path(node.get('href')).name
                    if not asset.is_file():
                        raise ValueError(f'Missing relocated asset: {asset}')
                    node.set('href', str(asset))
    write(root, args.output)
    print(f'Wrote {args.output}: current neural weights and conventional coefficients; fresh sampler/optimizer unless separately restored.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('handoff')
    stage.add_argument('pretrain', type=Path)
    stage.add_argument('energy', type=Path)
    stage.add_argument('--output', type=Path, required=True)
    stage.add_argument('--run', type=Path)
    stage.set_defaults(function=handoff)
    model = commands.add_parser('extract')
    model.add_argument('model', type=Path)
    model.add_argument('template', type=Path)
    model.add_argument('--output', type=Path, required=True)
    model.add_argument('--asset-dir', type=Path)
    model.add_argument('--pin-template-jastrows', action='store_true',
                       help='Explicit new-model association for a neural-only artifact; retain template J values.')
    model.set_defaults(function=extract)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
