#!/usr/bin/env python3
"""Validate a frozen qualification snapshot; print a recipe unless execution is explicit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def digest(path):
    """Hash even application binaries without retaining their entire image in memory."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main():
    """Resolve only two pinned recipes; never allocate resources or select a new policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='validate snapshot/schema/template only')
    parser.add_argument('--profile', choices=('SINGLE_GPU', 'CPU_MULTINODE'))
    for name in ('app', 'model', 'input-list', 'runtime-path-file', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--allocation', help='existing BESMS allocation, never submitted here')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    snapshot = json.loads((base / 'qualified_profiles.json').read_text())
    profiles = snapshot['profiles']
    assert snapshot['schema_version'] == 1 and set(profiles) == {'SINGLE_GPU', 'CPU_MULTINODE', 'HIP_COMPILE', 'PRODUCTION_PENDING'}
    for name in ('SINGLE_GPU', 'CPU_MULTINODE'):
        assert profiles[name]['status'] == 'PASS' and profiles[name]['runtime_qualified'] is True
        assert len(profiles[name]['source']) == 40 and len(profiles[name]['binary_sha256']) == 64 and profiles[name]['evidence']
        for field in ('source', 'binary_sha256'):
            int(profiles[name][field], 16)
    assert not profiles['HIP_COMPILE']['runtime_qualified'] and not profiles['PRODUCTION_PENDING']['runtime_qualified']
    assert profiles['HIP_COMPILE']['status'] == 'PRIOR_SOURCE_COMPILE_ONLY' and profiles['PRODUCTION_PENDING']['status'] == 'PENDING'
    assert len(profiles['CPU_MULTINODE']['input_sha256']) == 2
    template = base / profiles['SINGLE_GPU']['template']
    assert digest(template) == profiles['SINGLE_GPU']['template_sha256']
    tree = ET.parse(template)
    assert [int(g.get('size')) for g in tree.findall("particleset[@name='e']/group")] == [2, 2]
    assert tree.find('.//psiformer').get('parameters') == '@MODEL@'
    if args.check:
        assert not args.execute, '--check never executes'
        print('PASS source-specific snapshot/schema/XML; HIP compile-only and production pending remain distinct')
        return
    if not all((args.profile, args.app, args.output)):
        parser.error('--profile, --app and --output are required for a recipe')
    profile, app, output = profiles[args.profile], args.app.resolve(), args.output.resolve()
    binary_hash = digest(app) if app.is_file() else None
    qualified = binary_hash == profile['binary_sha256']
    controls = dict(OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1')
    inputs = []
    if args.profile == 'SINGLE_GPU':
        if not args.model or not args.model.is_file():
            parser.error('the protected matching LiH --model is required; no model is distributed')
        model = args.model.resolve()
        model_hash = digest(model)
        qualified = qualified and model_hash == profile['model_sha256']
        inputs = [model]
        tree.find('.//psiformer').set('parameters', str(model))
        controls.update(profile['environment'])
        command = [str(app), 'input.xml']
    else:
        if not args.input_list or not args.runtime_path_file or not args.allocation or not args.allocation.isdigit():
            parser.error('CPU recipe requires prepared --input-list, exact --runtime-path-file and existing --allocation')
        inputs = [args.input_list.resolve()] + [args.input_list.resolve().parent / f'slot{i}.xml' for i in (0, 1)]
        if not all(p.is_file() for p in inputs + [args.runtime_path_file]):
            parser.error('NOT_QUALIFIED: required prepared input/runtime file is missing')
        if digest(inputs[0]) != profile['list_sha256'] or [digest(p) for p in inputs[1:]] != profile['input_sha256']:
            parser.error('NOT_QUALIFIED: prepared CPU inputs do not match the frozen mixed-slot profile')
        if digest(args.runtime_path_file) != profile['runtime_path_sha256']:
            parser.error('NOT_QUALIFIED: runtime path digest mismatch; refusing to publish or use its contents')
        assert all([int(g.get('size')) for g in ET.parse(p).findall(".//particleset[@name='e']/group")] == [1, 1] for p in inputs[1:])
        controls['LD_LIBRARY_PATH'] = args.runtime_path_file.read_text().strip()
        command = ['srun', '--jobid=' + args.allocation, '--exclusive', '--exact', '--nodes=2', '--nodelist=' + profile['nodes'],
            '--ntasks=4', '--ntasks-per-node=2', '--cpus-per-task=1', '--distribution=cyclic:cyclic', '--cpu-bind=cores',
            '--mem=14G', '--time=00:01:00', '--kill-on-bad-exit=1', '--input=none', str(app), 'inputs.txt']
    command = ['timeout', '-k', '5', str(profile['seconds'])] + command
    report = dict(qualification='QUALIFIED_PIN' if qualified else 'NOT_QUALIFIED', execute=args.execute,
        qualified_source=profile['source'], binary_sha256=binary_hash, expected_binary_sha256=profile['binary_sha256'],
        evidence=profile['evidence'], command=command,
        environment=controls, output=str(output), input_sha256={str(p): digest(p) for p in inputs})
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return
    if not qualified:
        parser.error('NOT_QUALIFIED: binary or protected input/runtime pin mismatch; refusing execution')
    output.mkdir(parents=True, exist_ok=False)
    if args.profile == 'SINGLE_GPU':
        tree.write(output / 'input.xml', encoding='utf-8', xml_declaration=True)
    else:
        for source, name in zip(inputs, ('inputs.txt', 'slot0.xml', 'slot1.xml')):
            shutil.copyfile(source, output / name)
    (output / 'launch.json').write_text(json.dumps(report, indent=2) + '\n')
    environment = {k: v for k, v in os.environ.items() if not k.startswith('PSIFORMER_')}
    environment.update(controls)
    os.chdir(output)
    os.execvpe(command[0], command, environment)


if __name__ == '__main__':
    main()
