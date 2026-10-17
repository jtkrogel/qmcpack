#!/usr/bin/env python3
"""Inert-until-authorized AE DEVICE SR pilot, then explicit exact restart qualification."""
import argparse, csv, hashlib, json, math, os, re, runpy, signal, subprocess, time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'source', 'input', 'model', 'output'):
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--source-commit', required=True)
p.add_argument('--pilot-report', type=Path)
p.add_argument('--full-workflow', action='store_true', help='qualified bounded pilot plus exact split/resume')
a = p.parse_args()
if a.full_workflow and a.pilot_report:
    p.error('--full-workflow and --pilot-report are mutually exclusive')
signal.signal(signal.SIGTERM, signal.default_int_handler)
utility = runpy.run_path(str(a.source/'tests/molecules/LiH_ae_psiformer_gpu/check_resident_device_authority.py'))
sha, fields, selected, render = (utility[key] for key in ('sha','fields','selected','render'))
envelopes = runpy.run_path(str(a.source/'tests/molecules/LiH_ae_psiformer_gpu/check_rejected_full_reuse.py'))['envelopes']
a.output.mkdir(exist_ok=False)
identity = dict(source=a.source_commit, binary_sha256=sha(a.binary), input_sha256=sha(a.input), model_sha256=sha(a.model))
initial = fields(a.model)
assert initial['values'].size == 666 and np.isfinite(initial['values']).all()
np.testing.assert_array_equal(initial['psiformer_model/shape'].reshape(-1), [2,2,2,2,8,2,1])
assert int(np.asarray(initial['psiformer_model/seed']).reshape(-1)[0]) == 37
assert initial['psiformer_model/layout_fingerprint'].reshape(-1)[0] == b'02f6eef5b28ef98b'
assert hashlib.sha256(np.asarray(initial['values'],dtype='<f8').tobytes()).hexdigest() == \
    '5fb6c32a3a9f6f46a22afe595cc41ff165ef4646fc1d8b89a7b43a4a1720a7ed'
assert all(int(initial['psiformer_model/'+key].reshape(-1)[0]) == 0 for key in ('paired_complex','periodic'))
report = dict(status='FAIL', **identity, cases={})
deadline = time.monotonic() + (180 if a.pilot_report or a.full_workflow else 60)

def history(folder):
    with next(folder.glob('*.training_info.csv')).open() as stream:
        return list(csv.DictReader(stream))

def checkpoint(folder):
    pointer, = folder.glob('*.last_checkpoint.h5')
    return tuple(fields(path) for path in envelopes(pointer))

def run(name, iterations, restart=None):
    folder = a.output/name
    folder.mkdir()
    render(ET.parse(a.input), a.model.resolve(), folder/'input.xml', iterations, restart)
    tree = ET.parse(folder/'input.xml')
    training = tree.find('qmc/training')
    training.set('optimizer','sr')
    for node in list(training):
        if node.get('name') in ('adam_beta1','adam_beta2','epsilon','learning_rate'):
            training.remove(node)
    controls = dict(learning_rate='1e-4', damping='.3', maximum_damping_attempts=1,
                   krylov_relative_tolerance='1e-9', krylov_absolute_tolerance='1e-10',
                   krylov_maximum_iterations=20, krylov_maximum_operator_applications=24,
                   krylov_maximum_seconds=10)
    for key,value in controls.items():
        ET.SubElement(training,'parameter',name=key).text = str(value)
    tree.write(folder/'input.xml',encoding='utf-8',xml_declaration=True)
    env = {key:value for key,value in os.environ.items() if not key.startswith('PSIFORMER_GPU_')}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OMPI_MCA_memory='^patcher',OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,tcp',
        PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES=str(2*1024**3),PSIFORMER_GPU_EXTERNAL_RESERVE_BYTES=str(256*1024**2),
        PSIFORMER_GPU_METROPOLIS_BYTES=str(16*1024**2),PSIFORMER_GPU_STATUS_CHECKS='synchronous',
        PSIFORMER_GPU_VGL_TARGETS='1',PSIFORMER_GPU_TARGET_WORKSPACE_BYTES=str(256*1024**2),
        PSIFORMER_GPU_TARGET_SCALAR_WORK_BUDGET=str(10**9),PSIFORMER_GPU_RESIDENT_DEVICE_AUTHORITY='1',
        PSIFORMER_GPU_RESIDENT_ENERGY_SR_BYTES=str(64*1024**2),
        PSIFORMER_GPU_RESIDENT_ENERGY_SR_HOST_BYTES=str(1024**2),PSIFORMER_GPU_RESIDENT_ENERGY_SR_REUSE='on')
    started = time.monotonic()
    with (folder/'console.log').open('w') as stream:
        child = subprocess.Popen(['/usr/bin/time','-v','-o','resource_time.txt',str(a.binary.resolve()),'input.xml'],
            cwd=folder,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code = child.wait(timeout=max(.001,min(60,deadline-time.monotonic())))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL); child.wait()
            raise AssertionError('native SR exceeded child60s/whole-helper cap')
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
    report['cases'][name] = dict(returncode=code,elapsed_seconds=time.monotonic()-started,
                                input_sha256=sha(folder/'input.xml'),case_path=str(folder.resolve()))
    text = (folder/'console.log').read_text()
    assert code == 0 and 'NOT_RUN' not in text, text[-4000:]
    rows = history(folder)
    expected = list(range(1 if restart else 0,iterations))
    assert [int(row['iteration']) for row in rows] == expected
    assert all(row['optimizer']=='sr' and int(row['samples'])==2 and int(row['committed'])==1
               and int(row['parameter_version'])==int(row['iteration'])+1
               and all(math.isfinite(float(row[key])) for key in ('mean_energy','variance','gradient_norm'))
               and int(row['krylov_iterations'])>0 and int(row['krylov_operator_applications'])>0 for row in rows)
    terminal = list(re.finditer(r'Resident DEVICE SR terminal: source_version=(\d+), samples=(\d+), snapshot_device_bytes=(\d+)',text))
    windows = list(re.finditer(r'Resident DEVICE authority window:.*uploaded_bytes=(\d+).*downloaded_bytes=(\d+).*device_copied_bytes=(\d+)',text))
    assert len(terminal)==len(windows)==len(rows)
    assert all(tuple(map(int,item.groups()))==(int(row['sample_version']),2,8*666) for item,row in zip(terminal,rows))
    assert text.count('SR diagnostics route=resident_device ')==len(rows)
    assert 'Resident DEVICE Adam' not in text
    mirrors = list(re.finditer(r'cold HOST mirror refresh: attempted_bytes=(\d+), version=(\d+)',text))
    assert len(mirrors)==1 and mirrors[0].start()>windows[-1].end()
    assert tuple(map(int,mirrors[0].groups()))==(8*666,iterations)
    assert 'successor DEVICE=0' in text and 'cold HOST moments=0' in text
    records = checkpoint(folder)
    values = np.asarray(selected(records[0],'/model/values')).reshape(-1)
    assert values.size==666 and np.isfinite(values).all()
    model, = folder.glob('*.training/*.parameters.h5')
    np.testing.assert_array_equal(fields(model)['values'],values)
    report['cases'][name].update(maximum_rss_KiB=int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',(folder/'resource_time.txt').read_text())[1]),
        warm_transfer_windows=[list(map(int,item.groups())) for item in windows],
        scope='Cold-mirror fence ordering checked; no generic total-D2H heuristic or Adam-moment claim.')
    return folder,rows,records

try:
    if not a.pilot_report:
        folder,rows,records = run('continuous_two',2)
        assert not np.array_equal(np.asarray(selected(records[0],'/model/values')).reshape(-1),initial['values'])
        report.update(status='PASS_PILOT_ONLY',protected_pilot_hdf={str(path):sha(path) for path in folder.rglob('*.h5')})
        if a.full_workflow:
            a.pilot_report = a.output/'pilot_report.json'
            a.pilot_report.write_text(json.dumps(report,indent=2)+'\n')
    if a.pilot_report:
        pilot = json.loads(a.pilot_report.read_text())
        assert pilot['status']=='PASS_PILOT_ONLY' and all(pilot[key]==value for key,value in identity.items())
        assert all(sha(Path(path))==value for path,value in pilot['protected_pilot_hdf'].items())
        continuous = Path(pilot['cases']['continuous_two']['case_path'])
        split,split_rows,_ = run('split_one',1)
        protected = {str(path):sha(path) for path in split.rglob('*.h5')}
        pointer, = split.glob('*.last_checkpoint.h5')
        resumed,resumed_rows,actual = run('resumed_two',2,pointer)
        assert all(sha(Path(path))==value for path,value in protected.items())
        for left,right in zip(checkpoint(continuous),actual):
            assert left.keys()==right.keys()
            for key in left:
                np.testing.assert_array_equal(left[key],right[key],err_msg=key)
        archives = [fields(next(folder.glob('*.training/*.parameters.h5'))) for folder in (continuous,resumed)]
        assert archives[0].keys()==archives[1].keys()
        for key in archives[0]:
            np.testing.assert_array_equal(archives[0][key],archives[1][key],err_msg=key)
        for left,right in zip(history(continuous),split_rows+resumed_rows):
            assert left.keys()==right.keys()
            assert all(left[key]==right[key] for key in left if key not in {'sampling_seconds','total_seconds','resident_bytes'})
        report.update(status='PASS_EXACT_INITIALIZED_RESTART',pilot_report_sha256=sha(a.pilot_report))
    assert sha(a.model)==identity['model_sha256'] and sha(a.input)==identity['input_sha256']
finally:
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

