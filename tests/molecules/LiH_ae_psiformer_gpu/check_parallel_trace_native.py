#!/usr/bin/env python3
"""Tiny real/open CUDA parallel-trace equivalence and exact restart, not a benchmark."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
from check_resident_device_authority import fields, render, selected, sha
from check_rejected_full_reuse import envelopes


def main():
    """Reuse the protected numerical model; keep every native child inside one deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('app', 'model', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    args.app, args.model = args.app.resolve(), args.model.resolve()
    assert sha(args.model) == '12f7fcedfbdad544d2f7a2b4e5e4fac6b405ef0f04b88100ccc8c394d0f76321'
    fixture = Path(__file__).with_name('input.xml')
    protected = {str(path): sha(path) for path in (args.model, fixture)}
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='FAIL', app_sha256=sha(args.app), protected=protected, cases={}, benchmark=False)
    deadline, active = time.monotonic() + 120, None

    def stop(signum=None, frame=None):
        """Retire only this checker's owned process group, including on interruption."""
        if active is not None and active.poll() is None:
            try:
                os.killpg(active.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            active.wait()
        if signum is not None:
            raise SystemExit(f'interrupted by signal {signum}')

    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}

    def run(label, iterations, targets=1, parallel=True, restart=None):
        """Prepare exactly one walker/crowd; inference and derivative owners remain distinct."""
        directory = args.output / label
        directory.mkdir()
        render(ET.parse(fixture), args.model, directory / 'input.xml', iterations, restart)
        tree = ET.parse(directory / 'input.xml')
        qmc = tree.find('qmc')
        qmc.find("parameter[@name='walkers_per_rank']").text = '1'
        inference = ET.SubElement(qmc, 'inference', interval='1', last='yes', move='alle')
        for node in qmc.findall('parameter'):
            ET.SubElement(inference, 'parameter', name=node.get('name')).text = node.text
        tree.write(directory / 'input.xml')
        env = {k: v for k, v in os.environ.items() if not k.startswith('PSIFORMER_GPU_')}
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMPI_MCA_memory='^patcher',
            OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,tcp', PSIFORMER_GPU_STATUS_CHECKS='synchronous',
            PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES=str(2 * 1024**3),
            PSIFORMER_GPU_EXTERNAL_RESERVE_BYTES=str(256 * 1024**2),
            PSIFORMER_GPU_VGL_TARGETS=str(targets), PSIFORMER_GPU_TARGET_WORKSPACE_BYTES=str(256 * 1024**2),
            PSIFORMER_GPU_TARGET_SCALAR_WORK_BUDGET=str(10**9))
        if parallel:
            env['PSIFORMER_GPU_VGL_PARALLEL_TRACE_BYTES'] = str(1024**2)
        nonlocal active
        remaining = min(45, deadline - time.monotonic())
        assert remaining > 0, 'aggregate deadline exhausted'
        with (directory / 'console.log').open('w') as log:
            active = subprocess.Popen([str(args.app), 'input.xml'], cwd=directory, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                assert active.wait(timeout=remaining) == 0, directory
            finally:
                stop()
        assert 'numerical device tile completed: backend=cuda' in (directory / 'console.log').read_text()
        rows = list(csv.DictReader(next(directory.glob('*.training_info.csv')).open()))
        assert [int(r['iteration']) for r in rows] == list(range(1 if restart else 0, iterations))
        assert all(r['committed'] == '1' and int(r['parameter_version']) == int(r['iteration']) + 1 and
            float(r['gradient_norm']) > 0 and all(math.isfinite(float(r[k])) for k in
            ('mean_energy', 'variance', 'gradient_norm')) for r in rows)
        scalars = sorted(directory.glob('*.inference/*.scalar.dat'))
        assert len(scalars) == len(rows)
        physical = []
        for path in scalars:
            lines = path.read_text().splitlines()
            assert any('complete yes' in line for line in lines), path
            header = next(line.split()[1:] for line in lines if line.startswith('#') and 'BlockWeight' in line)
            values, = [line.split() for line in lines if line.strip() and not line.startswith('#')]
            assert len(values) == len(header), path
            row = {k: float(v) for k, v in zip(header, values) if k != 'BlockCPU'}
            assert row['BlockWeight'] == 1 and all(map(math.isfinite, row.values()))
            physical.append(row)
        latest, = directory.glob('*.last_checkpoint.h5')
        record = [fields(path) for path in envelopes(latest)]
        assert np.any(selected(record[0], '/model/values').reshape(-1) != fields(args.model)['values'].reshape(-1))
        report['cases'][label] = dict(input_sha256=sha(directory / 'input.xml'), targets=targets, parallel=parallel)
        return latest, rows, record, physical

    try:
        scalar, continuous, full = run('scalar', 1, parallel=False), run('parallel', 2), run('parallel_full', 1, targets=4)
        split = run('split', 1)
        saved = {str(p): sha(p) for p in split[0].parent.rglob('*.h5')}
        resumed = run('resumed', 2, restart=split[0])
        for route in (continuous, full):
            assert scalar[3][0].keys() == route[3][0].keys()
            np.testing.assert_allclose(list(scalar[3][0].values()), list(route[3][0].values()), rtol=2e-10, atol=2e-11)
            for key in ('mean_energy', 'variance', 'gradient_norm'):
                np.testing.assert_allclose(float(scalar[1][0][key]), float(route[1][0][key]), rtol=2e-10, atol=2e-11)
        for left, right in zip(continuous[2], resumed[2]):
            assert left.keys() == right.keys()
            for key in left:
                np.testing.assert_array_equal(left[key], right[key], err_msg=key)
        for left, right in zip(continuous[1], split[1] + resumed[1]):
            assert left.keys() == right.keys()
            assert all(left[k] == right[k] for k in left if k not in {'sampling_seconds', 'total_seconds', 'resident_bytes'})
        assert continuous[3][-1] == resumed[3][-1]
        assert saved == {p: sha(Path(p)) for p in saved}
        assert protected == {p: sha(Path(p)) for p in protected}
        report.update(status='PASS', exact_parallel_restart=True, rtol=2e-10, atol=2e-11,
                      scope='real/open FP64 B1 crowd1; no warm-allocation or throughput claim')
    finally:
        stop()
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        for number, handler in previous.items():
            signal.signal(number, handler)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
