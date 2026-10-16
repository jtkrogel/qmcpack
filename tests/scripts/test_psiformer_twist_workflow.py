#!/usr/bin/env python3
"""Deterministic external-workflow contracts; fake process, real scalar statistics."""
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('twist_workflow', ROOT / 'utils/psiformer_twist_workflow.py')
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)

XML = '''<simulation><project id="source" series="0"/><random seed="1"/>
<qmcsystem><simulationcell><parameter name="lattice">8 0 0 .8 8 0 .4 .6 8</parameter>
<parameter name="bconds">p p p</parameter></simulationcell>
<particleset name="ion0" size="1"><group name="H" size="1"><parameter name="charge">1</parameter>
<attrib name="position">0 0 0</attrib></group></particleset>
<particleset name="e"><group name="u" size="1"><parameter name="mass">1</parameter></group></particleset>
<wavefunction name="psi0" target="e"><psiformer system="all_electron" source="ion0"
initialization="psiformer_complex_periodic_v1" initialization_seed="17" feature_policy="periodic_torus_v1"
max_clone_numeric_bytes="16777216" optimize="no" gpu="cpu"/></wavefunction>
<hamiltonian name="h0" target="e"><pairpot type="coulomb" source="e" target="e"/>
<pairpot type="coulomb" source="ion0" target="ion0"/>
<pairpot type="coulomb" source="ion0" target="e"/></hamiltonian></qmcsystem>
<qmc method="vmc_batch"><parameter name="blocks">4</parameter></qmc></simulation>'''

FAKE = '''#!/usr/bin/env python3
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
seed = int(root.find('random').get('seed'))
print('fake ordinary input accepted', flush=True)
if seed == 99:
    sys.exit(3)
project = root.find('project').get('id')
series = len(root.findall('qmc'))-1
values = [seed-1, seed+1, seed-1, seed+1]
Path(f'{project}.s{series:03d}.scalar.dat').write_text('# index LocalEnergy BlockCPU\\n'+
    ''.join(f'{i} {value} .01\\n' for i,value in enumerate(values)))
'''


class IndependentTwistWorkflow(unittest.TestCase):
    """Keep workflow physics, provenance and failure contracts independent of QMCPACK runtime."""

    def setUp(self):
        """Prepare tiny inline XML and a process-shaped deterministic fake binary."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / 'source.xml'
        self.source.write_text(XML)
        self.binary = self.base / 'fake_qmcpack'
        self.binary.write_text(FAKE)
        self.binary.chmod(0o755)
        self.output = self.base / 'runs'
        self.manifest = {'version': 1, 'independent_streams': True, 'twists': [
            {'id': 'gamma', 'input': 'source.xml', 'fractional_twist': [0, 0, 0], 'weight': 1,
             'seed': 2, 'measurement_series': 0, 'discard_blocks': 0},
            {'id': 'nonzero', 'input': 'source.xml', 'fractional_twist': [.17, -.11, .07], 'weight': 3,
             'seed': 4, 'measurement_series': 0, 'discard_blocks': 0}]}

    def launch(self):
        """Exercise ordinary preparation, process launch and aggregation with exact written manifest."""
        path = self.base / 'twists.json'
        path.write_text(json.dumps(self.manifest))
        result = workflow.run_workflow(path, self.binary, self.output)
        return result, json.loads(result.read_text())

    def test_weighted_energy_statistics_and_fresh_attempts(self):
        """Normalize explicit weights and combine independent MC errors, never wavefunctions."""
        path, result = self.launch()
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['restart_supported'])
        self.assertEqual([entry['normalized_weight'] for entry in result['twists']], [.25, .75])
        self.assertEqual(result['mean_energy_hartree'], 3.5)
        self.assertAlmostEqual(result['mc_standard_error_hartree'], .5*math.sqrt(.25**2+.75**2))
        self.assertEqual(result['binary_sha256'], workflow.sha256(self.binary))
        for entry in result['twists']:
            self.assertEqual(entry['autocorrelation_factor'], 1)
            self.assertEqual(entry['blocks'], 4)
            root = workflow.ET.parse(entry['generated_input']).getroot()
            self.assertEqual(root.find('random').get('seed'), str(entry['seed']))
            self.assertEqual([float(v) for v in root.find('.//psiformer').get('twist').split()], entry['fractional_twist'])
            self.assertEqual(entry['generated_input_sha256'], workflow.sha256(Path(entry['generated_input'])))
            self.assertEqual(entry['initialization_profile'], 'psiformer_complex_periodic_v1')
            self.assertEqual(entry['initialization_seed'], '17')
            self.assertIn('fake ordinary input accepted', (Path(entry['generated_input']).parent/'qmcpack.log').read_text())
            self.assertEqual(entry['block_energy_variance'], 1)
            self.assertNotIn('variance', entry)
        second, repeated = self.launch()
        self.assertNotEqual(second.parent, path.parent)
        self.assertEqual(repeated['mean_energy_hartree'], result['mean_energy_hartree'])

    def test_skew_conversion_preserves_fractional_representative(self):
        """Every actual lattice row has phase kappa dot A_i=2*pi*t_i; no modulo folding."""
        rows = [[8, 0, 0], [.8, 8, 0], [.4, .6, 8]]
        twist = [1.17, -.11, .07]
        kappa = workflow.cartesian_twist(rows, twist)
        for row, fractional in zip(rows, twist):
            self.assertAlmostEqual(sum(a*b for a, b in zip(row, kappa)), 2*math.pi*fractional)
        shifted = workflow.cartesian_twist(rows, [.17, -.11, .07])
        self.assertNotEqual(kappa, shifted)
        with self.assertRaises(ValueError):
            workflow.cartesian_twist([[0, 0, 0]]*3, twist)

    def test_zero_weight_is_reported_and_allzero_rejected(self):
        """Zero-weight sectors remain explicit but contribute neither energy nor error."""
        self.manifest['twists'][0]['weight'] = 0
        _, result = self.launch()
        self.assertEqual(len(result['twists']), 2)
        self.assertEqual(result['mean_energy_hartree'], 4)
        self.assertEqual(result['mc_standard_error_hartree'], .5)
        self.manifest['twists'][1]['weight'] = 0
        with self.assertRaisesRegex(ValueError, 'sum must be positive'):
            self.launch()

    def test_invalid_manifest_does_not_launch_or_allocate_attempt(self):
        """Reject unsafe or correlated inputs before writing execution artifacts."""
        original = copy.deepcopy(self.manifest)
        mutations = [lambda m: m.update(independent_streams=False),
                     lambda m: m.update(restart='old-result.json'),
                     lambda m: m['twists'][0].update(weight=-1),
                     lambda m: m['twists'][0].update(weight=float('nan')),
                     lambda m: m['twists'][0].update(id='../escape'),
                     lambda m: m['twists'][1].update(seed=2),
                     lambda m: m['twists'][1].update(fractional_twist=[0, 0, 0]),
                     lambda m: m['twists'][0].update(measurement_series=1),
                     lambda m: m['twists'][0].update(checkpoint='foreign.h5')]
        for mutate in mutations:
            self.manifest = copy.deepcopy(original)
            mutate(self.manifest)
            with self.subTest(manifest=self.manifest), self.assertRaises(ValueError):
                self.launch()
            self.assertFalse(self.output.exists())

    def test_physical_mismatch_and_restart_import_rejected(self):
        """Keep fixed-N/mass/ion/cell identity and never silently accept imported walkers."""
        other = self.base / 'other.xml'
        other.write_text(XML.replace('name="mass">1', 'name="mass">2'))
        self.manifest['twists'][1]['input'] = 'other.xml'
        with self.assertRaisesRegex(ValueError, 'fixed-N'):
            self.launch()
        other.write_text(XML.replace('<qmc method=', '<mcwalkerset fileroot="foreign"/><qmc method='))
        with self.assertRaisesRegex(ValueError, 'Restart/walker|self-contained'):
            self.launch()
        other.write_text(XML.replace('</simulationcell>', '<parameter name="scale">2</parameter></simulationcell>'))
        with self.assertRaisesRegex(ValueError, 'unscaled lattice'):
            self.launch()
        self.assertFalse(self.output.exists())

    def test_imported_head_external_attributes_and_unphysical_hamiltonian_rejected(self):
        """Initialized-only input checks also cover real Gamma, unsafe associations and nonphysical terms."""
        bad_inputs = [XML.replace('source="ion0"\ninitialization=', 'source="ion0" parameters="weights.h5"\ninitialization='),
                      XML.replace('source="ion0"\ninitialization=', 'source="ion0" configuration="config.h5"\ninitialization='),
                      XML.replace('source="ion0"\ninitialization=', 'source="ion0" export_parameters="export.h5"\ninitialization='),
                      XML.replace('source="ion0"\ninitialization=', 'source="ion0" filename="weights.h5"\ninitialization='),
                      XML.replace('hamiltonian name="h0" target="e"', 'hamiltonian name="h0" target="other"'),
                      XML.replace('pairpot type="coulomb" source="e"', 'pairpot physical="no" type="coulomb" source="e"')]
        for source in bad_inputs:
            self.source.write_text(source)
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.launch()
            self.assertFalse(self.output.exists())
        self.source.write_text(bad_inputs[0].replace('psiformer_complex_periodic_v1', 'deepqmc_psiformer_v1'))
        with self.assertRaisesRegex(ValueError, 'parameter/configuration/export'):
            self.launch()

    def test_file_hash_uses_bounded_reads(self):
        """Executable/checkpoint hashing must not call whole-file read_bytes."""
        path = self.base/'large-artifact'
        payload = b'0123456789abcdef'*(2**17)
        path.write_bytes(payload)
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('unbounded file read')):
            self.assertEqual(workflow.sha256(path), hashlib.sha256(payload).hexdigest())

    def test_timeout_preserves_directly_streamed_partial_log(self):
        """A killed subprocess leaves diagnostic output on disk, never buffered in a PIPE."""
        self.binary.write_text('#!/usr/bin/env python3\nimport time\nprint("partial-before-timeout", flush=True)\ntime.sleep(10)\n')
        manifest = self.base/'twists.json'
        manifest.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            workflow.run_workflow(manifest, self.binary, self.output, timeout=1)
        result = next(self.output.glob('attempt_*/result.json'))
        self.assertEqual(json.loads(result.read_text())['status'], 'incomplete')
        self.assertIn('partial-before-timeout', (result.parent/'gamma/qmcpack.log').read_text())

    def test_gpu_training_and_unbounded_caps_rejected(self):
        """Admit initialized inference only on GPU and preserve both CPU numeric caps."""
        training = '<qmc method="wf_train"><training objective="energy" optimizer="adam"><parameter name="maximum_retained_bytes">10000</parameter><parameter name="checkpoint_path">x</parameter></training></qmc>'
        self.source.write_text(XML.replace('<qmc method="vmc_batch">', training+'<qmc method="vmc_batch">').replace('gpu="cpu"', 'gpu="cuda" max_gpu_numeric_bytes="10000"'))
        with self.assertRaisesRegex(ValueError, 'paired CPU'):
            self.launch()
        self.source.write_text(XML.replace('16777216', '0'))
        with self.assertRaisesRegex(ValueError, 'byte cap'):
            self.launch()
        self.assertFalse(self.output.exists())

    def test_partial_failure_keeps_evidence_and_is_never_complete(self):
        """A failed twist cannot produce an apparently valid weighted average or reused attempt."""
        self.manifest['twists'][1]['seed'] = 99
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            self.launch()
        results = list(self.output.glob('attempt_*/result.json'))
        self.assertEqual(len(results), 1)
        result = json.loads(results[0].read_text())
        self.assertEqual(result['status'], 'incomplete')
        self.assertNotIn('mean_energy_hartree', result)
        self.assertIn('fake ordinary input accepted', (results[0].parent/'nonzero/qmcpack.log').read_text())

    def test_measurement_series_discard_and_finite_contracts(self):
        """Read explicit fixed-P LocalEnergy only, with enough post-discard finite blocks."""
        self.manifest['twists'][0]['discard_blocks'] = 3
        with self.assertRaisesRegex(ValueError, 'at least two'):
            self.launch()
        scalar = self.base/'bad.scalar.dat'
        scalar.write_text('# index LocalEnergy\n0 nan\n1 1\n')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            workflow.measure(scalar, 0, workflow.load_statistics())


def create_family(directory, count=1, completed=1, generation='one'):
    """Write tiny selected-family metadata only; production C++ still validates real model/sampler payloads."""
    import h5py
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    manifest = directory/'checkpoint.manifest.h5'
    prefix = f'checkpoint.manifest.h5.generation-{generation}'
    with h5py.File(manifest, 'w') as archive:
        group = archive.create_group('wftrain_distributed_manifest')
        group['format_version'] = [1, 0, 0]
        group['participant_count'] = count
        group['rank_topology'] = [1, 1, 1, 1, 1]*count
        group['completion_cookie'] = [b'wftrain-distributed-manifest-complete-v1']
        group['generation'] = [generation.encode()]
        group['root_envelope_filename'] = [(prefix+'.root.h5').encode()]
        group['sampler_shard_prefix'] = [(prefix+'.sampler').encode()]
    with h5py.File(directory/(prefix+'.root.h5'), 'w') as archive:
        archive['wftrain_checkpoint/coordinator/completed_iterations'] = completed
        archive['wftrain_checkpoint/stage/completed_stage_iterations'] = completed
        archive['wftrain_checkpoint/stage/configuration_fingerprint'] = [('identity-'+directory.name).encode()]
    for rank in range(count):
        (directory/f'{prefix}.sampler.rank-{rank:08d}.h5').write_bytes(b'tiny synthetic sampler payload')
    return manifest


class SafeTwistRestart(unittest.TestCase):
    """Exercise complete selected families and real process-shaped restart flow without neural runtime."""

    launch = IndependentTwistWorkflow.launch

    def setUp(self):
        """Add a qualified-shaped CPU training stage and fake family publisher to existing tiny input."""
        IndependentTwistWorkflow.setUp(self)
        training = '<qmc method="wf_train"><training objective="energy" optimizer="adam"><parameter name="iterations">1</parameter><parameter name="maximum_retained_bytes">10000</parameter><parameter name="checkpoint_path">ignored.h5</parameter></training></qmc>'
        self.source.write_text(XML.replace('<qmc method="vmc_batch">', training+'<qmc method="vmc_batch">'))
        for entry in self.manifest['twists']:
            entry['measurement_series'] = 1
        self.binary.write_text(FAKE+inspect.getsource(create_family)+'''
import h5py
with (Path(__file__).parent/'launches').open('a') as stream:
    stream.write(project+'\\n')
training = root.find('qmc/training')
controls = {p.get('name'): p.text for p in training.findall('parameter')}
target = int(controls.get('iterations', '1'))
old = -1
if 'restart_path' in controls:
    for name in ('gamma', 'nonzero'):
        assert (Path.cwd().parent/name/'restart_family/checkpoint.manifest.h5').is_file()
    with h5py.File(controls['restart_path']) as f:
        filename = f['wftrain_distributed_manifest/root_envelope_filename'][()].item().decode()
    with h5py.File(Path(controls['restart_path']).parent/filename) as f:
        old = int(f['wftrain_checkpoint/coordinator/completed_iterations'][()])
if target > old:
    create_family(Path.cwd(), completed=target, generation=str(target))
''')

    def resume(self, prior, iterations=1):
        """Consume one explicit prior complete workflow result with unchanged manifest/binary."""
        return workflow.run_workflow(self.base/'twists.json', self.binary, self.output,
                                     restart_result=prior, iterations=iterations)

    def test_equal_target_snapshots_before_all_launches_and_extended_target(self):
        """Equal target reuses private input families; later updates select new published families."""
        prior, first = self.launch()
        original_hashes = [entry['checkpoint_family']['sha256'] for entry in first['twists']]
        resumed = self.resume(prior)
        second = json.loads(resumed.read_text())
        self.assertTrue(second['restart_supported'])
        self.assertEqual(second['restart_parent']['sha256'], workflow.sha256(prior))
        for index, entry in enumerate(second['twists']):
            self.assertEqual(Path(entry['checkpoint_family']['manifest']).parent.name, 'restart_family')
            self.assertEqual(entry['checkpoint_family']['sha256'], original_hashes[index])
            self.assertEqual(entry['checkpoint_artifact_sha256'], {})
            self.assertEqual(workflow.checkpoint_family(Path(first['twists'][index]['checkpoint_family']['manifest']), original_hashes[index])['sha256'], original_hashes[index])
        extended = json.loads(self.resume(resumed, iterations=2).read_text())
        for entry in extended['twists']:
            self.assertNotEqual(Path(entry['checkpoint_family']['manifest']).parent.name, 'restart_family')
            self.assertEqual(entry['checkpoint_family']['completed_iterations'], 2)

    def test_reject_corrupt_last_family_before_any_subprocess(self):
        """A valid first family cannot launch before a later family's complete snapshot admission."""
        prior, result = self.launch()
        launches = (self.base/'launches').read_bytes()
        family = result['twists'][1]['checkpoint_family']
        root = next(name for name in family['sha256'] if name.endswith('.root.h5'))
        (Path(family['manifest']).parent/root).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            self.resume(prior)
        self.assertEqual((self.base/'launches').read_bytes(), launches)

    def test_default_iteration_control_can_restore_without_other_override(self):
        """An initially omitted default limit may become explicit only as the requested restart target."""
        self.source.write_text(self.source.read_text().replace('<parameter name="iterations">1</parameter>', ''))
        prior, _ = self.launch()
        resumed = json.loads(self.resume(prior).read_text())
        for entry in resumed['twists']:
            root = workflow.ET.parse(entry['generated_input']).getroot()
            self.assertEqual(workflow.parameter(root.find('qmc/training'), 'iterations').text, '1')

    def test_reject_foreign_partial_target_template_and_ineligible_result(self):
        """All prior workflow identity/admission failures precede process launch."""
        prior, original = self.launch()
        launches = (self.base/'launches').read_bytes()
        for mutate in (lambda r: r.update(status='incomplete'), lambda r: r.update(binary_sha256='foreign'),
                       lambda r: r['twists'][0].update(seed=99),
                       lambda r: r['twists'][0].update(initialization_profile='deepqmc_psiformer_v1'),
                       lambda r: r['twists'][0].update(input_sha256='foreign'),
                       lambda r: r['twists'][0].update(checkpoint_family=r['twists'][1]['checkpoint_family'])):
            result = copy.deepcopy(original)
            mutate(result)
            prior.write_text(json.dumps(result))
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.resume(prior)
        prior.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, 'target below'):
            self.resume(prior, iterations=0)
        self.assertEqual((self.base/'launches').read_bytes(), launches)

    def test_selected_multirank_family_paths_missing_progress_and_copy(self):
        """Select exact named shards, reject traversal/missing data, and copy without sharing inodes."""
        import h5py
        manifest = create_family(self.base/'family', count=2)
        hashes = {p.name: workflow.sha256(p) for p in manifest.parent.iterdir()}
        family = workflow.checkpoint_family(manifest, hashes)
        self.assertEqual(len(family['sha256']), 4)
        snapshot = workflow.snapshot_family(family, self.base/'snapshot')
        self.assertEqual(snapshot['sha256'], hashes)
        for name in hashes:
            self.assertNotEqual((manifest.parent/name).stat().st_ino, (self.base/'snapshot'/name).stat().st_ino)
        shard = next(path for path in (self.base/'snapshot').iterdir() if 'rank-00000001' in path.name)
        shard.unlink()
        with self.assertRaises(OSError):
            workflow.checkpoint_family(Path(snapshot['manifest']), hashes)
        with h5py.File(manifest.parent/next(name for name in hashes if name.endswith('.root.h5')), 'r+') as archive:
            archive['wftrain_checkpoint/stage/completed_stage_iterations'][()] = 2
        progress_hashes = {p.name: workflow.sha256(p) for p in manifest.parent.iterdir()}
        with self.assertRaisesRegex(ValueError, 'counts disagree'):
            workflow.checkpoint_family(manifest, progress_hashes)
        with h5py.File(manifest, 'r+') as archive:
            del archive['wftrain_distributed_manifest/root_envelope_filename']
            archive['wftrain_distributed_manifest/root_envelope_filename'] = [b'../foreign.root.h5']
        hashes[manifest.name] = workflow.sha256(manifest)
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            workflow.checkpoint_family(manifest, hashes)
        manifest = create_family(manifest.parent, count=2)
        with h5py.File(manifest, 'r+') as archive:
            archive['wftrain_distributed_manifest/completion_cookie'][0] = b'incomplete'
        hashes = {p.name: workflow.sha256(p) for p in manifest.parent.iterdir()}
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            workflow.checkpoint_family(manifest, hashes)


if __name__ == '__main__':
    unittest.main()
