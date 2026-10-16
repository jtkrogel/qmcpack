#!/usr/bin/env python3
"""One actual CPU wrapper train1/equal-restore->VMC gate, not four trajectories."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def load_module(name, path):
    """Reuse existing checker/template helpers without a second XML authoring framework."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """Require wrapper snapshots, zero-update restore and identical following physical scalar series."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    workflow = load_module('twist_workflow', here.parents[2]/'utils/psiformer_twist_workflow.py')
    checker = load_module('existing_training_checker', here/'check_cpu_training.py')
    args.output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='restart_', dir=args.output.resolve()))
    template = checker.input_xml(here/'complex_twist.xml', work/'template', 'adam', 1)
    tree = ET.parse(template)
    vmc = ET.parse(here/'complex_twist.xml').getroot().find('qmc')
    workflow.parameter(vmc, 'walkers_per_rank').text = '1'
    workflow.parameter(vmc, 'warmupSteps').text = '0'
    tree.getroot().append(vmc)
    tree.write(template, encoding='unicode', xml_declaration=True)
    manifest = work/'twists.json'
    manifest.write_text(json.dumps({'version': 1, 'independent_streams': True, 'twists': [
        {'id': 'nonzero', 'input': str(template), 'fractional_twist': [.17, -.11, .07],
         'weight': 1, 'seed': 202610202, 'measurement_series': 1, 'discard_blocks': 0}]}))
    first_path = workflow.run_workflow(manifest, args.binary, work/'runs')
    first = json.loads(first_path.read_text())['twists'][0]
    family = first['checkpoint_family']
    resumed_path = workflow.run_workflow(manifest, args.binary, work/'runs',
                                         restart_result=first_path, iterations=1)
    resumed = json.loads(resumed_path.read_text())['twists'][0]
    if workflow.checkpoint_family(Path(family['manifest']), family['sha256']) != family:
        raise RuntimeError('Source checkpoint family changed')
    if (resumed['checkpoint_artifact_sha256'] or resumed['checkpoint_family']['sha256'] != family['sha256'] or
            Path(resumed['checkpoint_family']['manifest']).parent.name != 'restart_family'):
        raise RuntimeError('Equal target must retain its private input family, without a new generation')
    directory = Path(resumed['generated_input']).parent
    console = (directory/'qmcpack.log').read_text()
    if ('WFTrain restored 1 completed update(s)' not in console or
            'WFTrain completed 0 block(s) and 0 committed update(s), 1 completed iteration(s)' not in console):
        raise RuntimeError('Expected exact zero-work restored training handoff')
    histories = list(directory.glob('*.training.csv'))
    if len(histories) != 1:
        raise RuntimeError('Expected one restored training history')
    with histories[0].open() as stream:
        rows = list(csv.DictReader(stream))
    if rows:
        raise RuntimeError('Restore unexpectedly wrote optimizer rows')
    physical = []
    for entry in (first, resumed):
        lines = Path(entry['scalar_file']).read_text().splitlines()
        columns = lines[0].split()[1:]
        physical.append([[float(value) for value, name in zip(row.split(), columns) if name != 'BlockCPU']
                         for row in lines[1:] if row.strip() and not row.lstrip().startswith('#')])
    if physical[0] != physical[1] or len(physical[0]) != 2:
        raise RuntimeError('Following VMC physical/sampler sequence differs (clock columns excluded)')
    print(f'PASS wrapper train1/equal-restore1->VMC: no optimizer work, private reusable family, '
          f'{sum(map(len, physical[0]))} identical nonclock scalars; evidence {work}')


if __name__ == '__main__':
    main()
