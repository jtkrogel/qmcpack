#!/usr/bin/env python3
"""Deterministic external-workflow contracts; fake process, real scalar statistics."""
import copy
import hashlib
import importlib.util
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


if __name__ == '__main__':
    unittest.main()
