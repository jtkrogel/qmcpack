#!/usr/bin/env python3
"""Tiny real/open DEVICE Adam authority, cold artifacts and exact native restart.

The supplied protected REAL model must match the supplied AE physical XML. The
ordinary CUDA Adam route is a HOST-authoritative, same-sample tolerance oracle;
the new route's split restart is exact. This is not a throughput benchmark.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(path):
    result = {}
    with h5py.File(path) as archive:
        archive.visititems(lambda name, item: result.update({name: item[()]})
                           if isinstance(item, h5py.Dataset) else None)
    return result


def selected(record, suffix):
    matches = [value for key, value in record.items() if key.endswith(suffix)]
    assert len(matches) == 1, (suffix, len(matches))
    return matches[0]


def validate_warm_transfers(windows, count, expected_d2h):
    assert all(int(upload) < 8 * count and int(download) == expected_d2h and int(copied) == 16 * count + 384
               for upload, download, copied in windows), windows


def render(template, model, destination, iterations, restart=None):
    tree = copy.deepcopy(template)
    root = tree.getroot()
    root.find('project').set('id', 'resident_authority')
    head = root.find('wavefunction/psiformer')
    for key in ('initialization', 'initialization_seed', 'configuration', 'parameter_checkpoint', 'optimize_indices'):
        head.attrib.pop(key, None)
    head.attrib.update(parameters=str(model), gpu='cuda', optimize='yes', optimize_scope='all')
    for node in list(root):
        if node.tag in ('qmc', 'walkerlogs'):
            root.remove(node)
    qmc = ET.SubElement(root, 'qmc', method='wf_train', move='alle', checkpoint='-1')
    training = ET.SubElement(qmc, 'training', objective='energy', optimizer='adam')
    def parameters(parent, values):
        for name, value in values.items():
            ET.SubElement(parent, 'parameter', name=name).text = str(value)
    parameters(training, dict(iterations=iterations, warmupsteps=0, learning_rate='1e-5',
        adam_beta1='.9', adam_beta2='.999', epsilon='1e-8', parameter_chunk_size=256,
        checkpoint_interval=1000, maximum_retained_bytes=512 * 1024**2))
    if restart:
        parameters(training, dict(restart_path=str(restart)))
    parameters(qmc, dict(walkers_per_rank=2, crowds=1, blocks=1, steps=1, warmupsteps=0,
        substeps=1, timestep='1e-8', usedrift='no', blocks_between_recompute=100))
    tree.write(destination, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    parser.add_argument('--model', type=Path, required=True, help='Initial REAL/open exported model from the numerical gate')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--input', type=Path, help='Exact matching physical XML; defaults to existing LiH AE fixture')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device-bytes', type=int, default=3 * 1024**3)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--rtol', type=float, default=2e-10)
    parser.add_argument('--atol', type=float, default=2e-11)
    parser.add_argument('--h5diff', default=shutil.which('h5diff'))
    args = parser.parse_args()
    if (not math.isfinite(args.timeout) or args.timeout <= 0 or args.device_bytes <= 0 or not args.h5diff or
            any(not math.isfinite(x) or x < 0 for x in (args.rtol, args.atol))):
        parser.error('positive finite timeout/device cap, finite nonnegative tolerances and h5diff are required')
    args.app, args.model, args.source = args.app.resolve(), args.model.resolve(), args.source.resolve()
    fixture = (args.input or args.source / 'tests/molecules/LiH_ae_psiformer_gpu/input.xml').resolve()
    helper = args.source / 'tests/molecules/LiH_ae_psiformer_gpu/check_rejected_full_reuse.py'
    envelopes = runpy.run_path(str(helper))['envelopes']
    protected = {str(path): sha(path) for path in (fixture, args.model, helper)}
    initial = fields(args.model)
    count = np.asarray(initial['values']).size
    # This is the protected numerical fixture, not a generic transfer allowance.
    assert count == 666 and np.isfinite(initial['values']).all()
    np.testing.assert_array_equal(np.asarray(initial['psiformer_model/shape']).reshape(-1), [2, 2, 2, 2, 8, 2, 1])
    for key in ('paired_complex', 'periodic'):
        assert int(np.asarray(initial['psiformer_model/' + key]).reshape(-1)[0]) == 0
    def text(value):
        item = np.asarray(value).reshape(-1)[0]
        return item.decode().rstrip('\0') if isinstance(item, bytes) else str(item)
    assert text(initial['psiformer_model/precision']) == 'fp64_reference'
    assert text(initial['psiformer_model/system']) == 'all_electron'
    assert text(initial['psiformer_model/feature_policy']) == 'open_euclidean_v1'
    assert int(np.asarray(initial['psiformer_model/seed']).reshape(-1)[0]) == 37
    np.testing.assert_array_equal(np.asarray(initial['psiformer_model/nuclear_charges']).reshape(-1), [3, 1])
    np.testing.assert_array_equal(np.asarray(initial['psiformer_model/nuclear_positions']).reshape(-1),
                                 [0, 0, 0, 3.0154361724853516, 0, 0])
    original = ET.parse(fixture)
    if args.output:
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=False)
    else:
        args.output = Path(tempfile.mkdtemp(prefix='psiformer_resident_authority_'))
    report = dict(status='FAIL', app_sha256=sha(args.app), protected=protected, canonical_count=count,
                  harness_sha256=sha(Path(__file__).resolve()), device_cap=args.device_bytes, benchmark=False, cases={})
    deadline = time.monotonic() + args.timeout
    def execute(command, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Resident native suite deadline exceeded')
        subprocess.run(command, timeout=min(180, remaining), check=True, **kwargs)
    def run(name, iterations, authority=True, restart=None):
        directory = args.output / name
        directory.mkdir()
        render(original, args.model, directory / 'input.xml', iterations, restart)
        env = {key: value for key, value in os.environ.items() if not key.startswith('PSIFORMER_GPU_')}
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMPI_MCA_memory='^patcher',
            OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,tcp',
            PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES=str(args.device_bytes),
            PSIFORMER_GPU_EXTERNAL_RESERVE_BYTES=str(256 * 1024**2),
            PSIFORMER_GPU_METROPOLIS_BYTES=str(16 * 1024**2), PSIFORMER_GPU_ADAM_BYTES=str(16 * 1024**2),
            PSIFORMER_GPU_STATUS_CHECKS='synchronous',
            PSIFORMER_GPU_VGL_TARGETS='1', PSIFORMER_GPU_TARGET_WORKSPACE_BYTES=str(256 * 1024**2),
            PSIFORMER_GPU_TARGET_SCALAR_WORK_BUDGET=str(10**9))
        if authority:
            env.update(PSIFORMER_GPU_RESIDENT_DEVICE_AUTHORITY='1',
                       PSIFORMER_GPU_RESIDENT_ENERGY_ADAM_BYTES=str(16 * 1024**2))
        with (directory / 'console.log').open('w') as log:
            execute([str(args.app), 'input.xml'], cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
        with next(directory.glob('*.training_info.csv')).open() as stream:
            rows = list(csv.DictReader(stream))
        expected = list(range(1 if restart else 0, iterations))
        assert [int(row['iteration']) for row in rows] == expected
        assert all(row['committed'] == '1' and row['optimizer'] == 'adam' and
            int(row['parameter_version']) == int(row['iteration']) + 1 and
            float(row['gradient_norm']) > 0 and
            all(math.isfinite(float(row[key])) for key in ('mean_energy', 'variance', 'gradient_norm'))
            for row in rows)
        latest, = directory.glob('*.last_checkpoint.h5')
        root, shard = envelopes(latest)
        record, sampler = fields(root), fields(shard)
        assert int(np.asarray(selected(record, '/accepted_update_count')).reshape(-1)[0]) == iterations
        model, = directory.glob('*.training/*.parameters.h5')
        np.testing.assert_array_equal(fields(model)['values'], selected(record, '/model/values'))
        report['cases'][name] = dict(input_sha256=sha(directory / 'input.xml'), iterations=iterations,
                                    authority=authority, cold_export_matches_checkpoint=True)
        return latest, rows, record, sampler, directory
    try:
        continuous = run('continuous_three', 3)
        split = run('split_one', 1)
        protected_split = {str(path): sha(path) for path in split[4].rglob('*.h5')}
        resumed = run('resumed_three', 3, restart=split[0])
        assert protected_split == {str(path): sha(Path(path)) for path in protected_split}
        left, right = envelopes(continuous[0]), envelopes(resumed[0])
        assert len(left) == len(right) == 2
        for a, b in zip(left, right):
            execute([args.h5diff, str(a), str(b)])
        joined = split[1] + resumed[1]
        assert len(joined) == len(continuous[1]) == 3
        ignored = {'sampling_seconds', 'total_seconds', 'resident_bytes'}
        for expected, actual in zip(continuous[1], joined):
            assert expected.keys() == actual.keys()
            assert all(expected[key] == actual[key] for key in expected if key not in ignored)
        host = run('host_oracle_three', 3, authority=False)
        # Cross-route reduction order can round differently; do not require exact
        # HOST/GPU trajectories, nor relax these declared tolerances after a failure.
        for suffix in ('/model/values', '/first_moment', '/second_moment'):
            np.testing.assert_allclose(selected(continuous[2], suffix), selected(host[2], suffix),
                                       rtol=args.rtol, atol=args.atol, err_msg=suffix)
        for suffix in ('/positions', '/rng_state', '/rng_sizes', '/global_step'):
            np.testing.assert_array_equal(selected(continuous[3], suffix), selected(host[3], suffix))
        assert len(continuous[1]) == len(host[1]) == 3
        for expected, actual in zip(continuous[1], host[1]):
            for key in ('mean_energy', 'variance', 'gradient_norm'):
                assert math.isclose(float(expected[key]), float(actual[key]), rel_tol=args.rtol, abs_tol=args.atol)
        assert not np.array_equal(selected(continuous[2], '/model/values').reshape(-1), initial['values'].reshape(-1))
        # Counter format is supplied by the component's fixed existing-metrics
        # window hook; it includes sampling, producer, proposal and FULL refresh.
        log = (continuous[4] / 'console.log').read_text()
        window_matches = list(re.finditer(r'Resident DEVICE authority window:.*uploaded_bytes=(\d+).*downloaded_bytes=(\d+).*device_copied_bytes=(\d+)', log))
        windows = [match.groups() for match in window_matches]
        assert len(windows) == 3, ('Expected all three complete native warm windows', windows)
        # Source-linked native_transfer_accounting_B.rst: K1 targets electrons,
        # not Cartesian directions. Synchronous small statuses exceed 8P for
        # this tiny model; require their exact sum, never a widened heuristic.
        small_d2h = dict(two_full=4720, sampler=924, c3_samples=1176, c3_finite=16,
                         adam_norms_status=2 * (16 * math.ceil(count / 128)) + 16, successor_cusps=16)
        expected_d2h = sum(small_d2h.values())
        assert expected_d2h == 7060
        validate_warm_transfers(windows, count, expected_d2h)
        mirrors = list(re.finditer(r'cold HOST mirror refresh: attempted_bytes=(\d+), version=(\d+)', log))
        assert len(mirrors) == 1 and mirrors[0].start() > window_matches[-1].end()
        assert tuple(map(int, mirrors[0].groups())) == (8 * count, 3)
        assert protected == {path: sha(Path(path)) for path in protected}
        report.update(status='PASS', exact_restart=True, host_oracle_rtol=args.rtol, host_oracle_atol=args.atol,
                      warm_transfer_windows=[dict(uploaded=int(u), downloaded=int(d), device_copied=int(c))
                                             for u, d, c in windows],
                      accounted_small_d2h=small_d2h, cold_mirror_bytes=8 * count,
                      cold_reader_gate='Separately required numerical cached-CPU-reader test',
                      profile='REAL/open AE ENERGY Adam MPI1/crowd1/W2/block1/step1 movealle/nodrift; no controls')
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS:', json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
