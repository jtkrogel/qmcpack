#!/usr/bin/env python3
"""Tiny initialized-only paired CPU/explicit-target CUDA XML scalar parity."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def main():
    """Keep matched XML/seed/phase, ignore clock fields and require the selected numerical route."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('existing_gpu_scalar_reader',
                                                here.parent/'LiH_ae_psiformer_gpu/compare_xml.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    args.output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='target_', dir=args.output.resolve()))
    results = {}
    for backend in ('cpu', 'cuda'):
        directory = work/backend
        directory.mkdir()
        tree = ET.parse(args.input)
        head = tree.getroot().find('.//psiformer')
        head.set('gpu', backend)
        if backend == 'cuda':
            head.set('max_gpu_numeric_bytes', str(64*1024*1024))
        tree.write(directory/'input.xml', encoding='unicode', xml_declaration=True)
        # Isolate each subprocess from unrelated developer target/optimizer controls.
        environment = dict(os.environ)
        for name in ('VGL_TARGETS', 'TARGET_WORKSPACE_BYTES', 'TARGET_SCALAR_WORK_BUDGET', 'TARGET_SELECTION',
                     'KINETIC_TARGETS', 'LIFTED_CHANNELS', 'RETAINED_COORDINATE_BYTES', 'METROPOLIS_BYTES',
                     'METROPOLIS_DRIFT_BYTES', 'ADAM_BYTES', 'RESIDENT_ENERGY_ADAM_BYTES'):
            environment.pop('PSIFORMER_GPU_'+name, None)
        if backend == 'cuda':
            environment.update(PSIFORMER_GPU_VGL_TARGETS='3', PSIFORMER_GPU_TARGET_WORKSPACE_BYTES=str(64*1024*1024),
                               PSIFORMER_GPU_TARGET_SCALAR_WORK_BUDGET='1000000000')
        with (directory/'qmcpack.log').open('w') as console:
            completed = subprocess.run([str(args.binary.resolve()), 'input.xml'], cwd=directory, env=environment,
                                       stdout=console, stderr=subprocess.STDOUT, timeout=150)
        if completed.returncode:
            raise RuntimeError(f'{backend} XML failed; evidence {directory}')
        if backend == 'cuda' and 'mode=EXPLICIT_TARGET, K=3' not in (directory/'qmcpack.log').read_text():
            raise RuntimeError('Strict CUDA input did not prepare the explicit paired target lane')
        results[backend] = reader.scalar_data(directory/'complex_cpu.s000.scalar.dat')
    if results['cpu'][0] != results['cuda'][0] or len(results['cpu'][1]) != len(results['cuda'][1]):
        raise RuntimeError('CPU/CUDA scalar schemas differ')
    maximum = 0.0
    for cpu, gpu in zip(results['cpu'][1], results['cuda'][1]):
        for left, right in zip(cpu, gpu):
            if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(left, right, rel_tol=2e-6, abs_tol=2e-6):
                raise RuntimeError(f'Paired CPU/target scalar mismatch: {left} versus {right}; evidence {work}')
            maximum = max(maximum, abs(left-right))
    print(f'PASS paired CPU/CUDA target XML: max nonclock difference {maximum:.12g}; evidence {work}')


if __name__ == '__main__':
    main()
