#!/usr/bin/env python3
"""Existing LiH BFD W2 DEVICE SR pilot, then exact initialized restart."""
import argparse, csv, hashlib, json, math, os, re, runpy, signal, subprocess, sys, tempfile, time
from types import SimpleNamespace
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'source', 'input'):
    p.add_argument('--'+name, type=Path, required=True)
outputs = p.add_mutually_exclusive_group(required=True)
outputs.add_argument('--output', type=Path, help='new manual evidence directory')
outputs.add_argument('--output-parent', type=Path, help='retain a unique evidence subdirectory per invocation')
p.add_argument('--source-commit', required=True)
p.add_argument('--pilot-report', type=Path)
p.add_argument('--full-workflow', action='store_true', help='qualified bounded pilot plus exact split/resume')
a = p.parse_args()
if a.full_workflow and a.pilot_report:
    p.error('--full-workflow and --pilot-report are mutually exclusive')
signal.signal(signal.SIGTERM, signal.default_int_handler)

def output_directory(output, parent):
    if parent:
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix='run-', dir=parent))
    output.mkdir(exist_ok=False)
    return output

utility = runpy.run_path(str(a.source/'tests/molecules/LiH_ae_psiformer_gpu/check_resident_device_authority.py'))
sha, fields, selected = (utility[key] for key in ('sha','fields','selected'))
sys.path.insert(0, str(a.source/'tests/molecules/LiH_pp_psiformer_wftrain'))
ecp = runpy.run_path(str(a.source/'tests/molecules/LiH_pp_psiformer_wftrain/check_gpu_ecp_replay.py'))
envelopes = runpy.run_path(str(a.source/'tests/molecules/LiH_ae_psiformer_gpu/check_rejected_full_reuse.py'))['envelopes']
a.output = output_directory(a.output, a.output_parent)
protected = [a.input, Path(__file__), a.source/'tests/molecules/LiH_pp_psiformer_wftrain/check_gpu_ecp_replay.py']
protected += [a.source/f'tests/molecules/LiH_pp/{element}.BFD.xml' for element in ('Li','H')]
identity = dict(source=a.source_commit, binary_sha256=sha(a.binary), protected={str(path):sha(path) for path in protected})
template = ET.parse(a.input)
head = template.find('wavefunction/psiformer')
assert head.get('initialization_seed')=='37' and head.get('system')=='pseudopotential'
assert template.find('random').get('seed')=='91427'
assert template.find("qmc/training/parameter[@name='learning_rate']").text.strip()=='1e-10'
p_count = 1593857 # Existing all-active internal model, not a new export/generator.
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
    for element in ('Li','H'):
        (folder/f'{element}.BFD.xml').symlink_to((a.source/f'tests/molecules/LiH_pp/{element}.BFD.xml').resolve())
    controls = SimpleNamespace(resident_ecp_profile=True, walkers=2, blocks=1)
    ecp['render'](a.input,folder/'input.xml','resident_ecp_sr','cuda',iterations,controls,restart)
    tree = ET.parse(folder/'input.xml')
    training = tree.find('qmc/training')
    training.set('optimizer','sr')
    for node in list(training):
        if node.get('name') in ('adam_beta1','adam_beta2','epsilon','learning_rate'):
            training.remove(node)
    controls = dict(learning_rate='1e-10', damping='.3', maximum_damping_attempts=1,
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
        PSIFORMER_GPU_RESIDENT_ENERGY_SR_BYTES=str(256*1024**2),
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
    assert all(tuple(map(int,item.groups()))==(int(row['sample_version']),2,8*p_count) for item,row in zip(terminal,rows))
    assert text.count('SR diagnostics route=resident_device ')==len(rows)
    assert 'Resident DEVICE Adam' not in text
    assert 'parameters=1593857, active=1593857' in text # All trainable; no native frozen subset claimed.
    mirrors = list(re.finditer(r'cold HOST mirror refresh: attempted_bytes=(\d+), version=(\d+)',text))
    assert len(mirrors)==1 and mirrors[0].start()>windows[-1].end()
    assert tuple(map(int,mirrors[0].groups()))==(8*p_count,iterations)
    assert 'successor DEVICE=0' in text and 'cold HOST moments=0' in text
    records = checkpoint(folder)
    values = np.asarray(selected(records[0],'/model/values')).reshape(-1)
    assert values.size==p_count and np.isfinite(values).all()
    assert all(float(row['variance'])>0 for row in rows), 'W2 must have distinct energies'
    appends = re.findall(r'Resident ENERGY accepted-grid ECP appended: points=24, prepared_score_capacity=4, weighted_virtual_tiles=6, weighted_virtual_configurations=24, last_virtual_tile_count=4, final_parameter_download_bytes=0',text)
    assert len(appends)==len(rows)
    scales = [float(value) for value in re.findall(r'WFTrain iteration \d+.*? displacement_norm ([^\s]+)',text)]
    assert len(scales)==len(rows) and all(math.isfinite(value) and value>0 for value in scales)
    model, = folder.glob('*.training/*.parameters.h5')
    np.testing.assert_array_equal(fields(model)['values'],values)
    report['cases'][name].update(maximum_rss_KiB=int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',(folder/'resource_time.txt').read_text())[1]),
        warm_transfer_windows=[list(map(int,item.groups())) for item in windows],
        displacement_norms=scales,scope='Same BFD W2 physics; actual bit-change proof from version1/2 exports in split stage; no convergence claim.')
    return folder,rows,records

try:
    if not a.pilot_report:
        folder,rows,records = run('continuous_two',2)
        report.update(status='PASS_PILOT_ONLY',protected_pilot_hdf={str(path):sha(path) for path in folder.rglob('*.h5')})
        if a.full_workflow:
            a.pilot_report = a.output/'pilot_report.json'
            a.pilot_report.write_text(json.dumps(report,indent=2)+'\n')
    if a.pilot_report:
        pilot = json.loads(a.pilot_report.read_text())
        assert pilot['status']=='PASS_PILOT_ONLY' and all(pilot[key]==value for key,value in identity.items())
        assert all(sha(Path(path))==value for path,value in pilot['protected_pilot_hdf'].items())
        continuous = Path(pilot['cases']['continuous_two']['case_path'])
        split,split_rows,first_state = run('split_one',1)
        protected = {str(path):sha(path) for path in split.rglob('*.h5')}
        pointer, = split.glob('*.last_checkpoint.h5')
        resumed,resumed_rows,actual = run('resumed_two',2,pointer)
        assert all(sha(Path(path))==value for path,value in protected.items())
        for left,right in zip(checkpoint(continuous),actual):
            assert left.keys()==right.keys()
            for key in left:
                np.testing.assert_array_equal(left[key],right[key],err_msg=key)
        assert not np.array_equal(np.asarray(selected(first_state[0],'/model/values')),
                                  np.asarray(selected(actual[0],'/model/values'))), 'No accepted trainable parameter bit changed'
        archives = [fields(next(folder.glob('*.training/*.parameters.h5'))) for folder in (continuous,resumed)]
        assert archives[0].keys()==archives[1].keys()
        for key in archives[0]:
            np.testing.assert_array_equal(archives[0][key],archives[1][key],err_msg=key)
        for left,right in zip(history(continuous),split_rows+resumed_rows):
            assert left.keys()==right.keys()
            assert all(left[key]==right[key] for key in left if key not in {'sampling_seconds','total_seconds','resident_bytes'})
        report.update(status='PASS_EXACT_INITIALIZED_RESTART',pilot_report_sha256=sha(a.pilot_report))
    assert all(sha(Path(path))==value for path,value in identity['protected'].items())
finally:
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

