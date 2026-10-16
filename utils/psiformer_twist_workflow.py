#!/usr/bin/env python3
"""Independent fixed-N twists: fresh ordinary XML jobs and weighted VMC energies.

This first cut never resumes/reuses an attempt or imports a checkpoint. Two-block
smokes are execution checks, not MC precision or twist-grid convergence evidence.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def sha256(path):
    """Identify retained input, executable and output bytes without mutable model hashes."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite(values):
    """Require finite numeric metadata before writing an attempt or launching a job."""
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite twist, lattice, weight or scalar value")


def cartesian_twist(rows, twist):
    """Solve A*kappa=2*pi*t for XML lattice ROW vectors, including skew cells."""
    def cross(a, b):
        """Return an ordinary three-dimensional cross product."""
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

    reciprocal = [cross(rows[1], rows[2]), cross(rows[2], rows[0]), cross(rows[0], rows[1])]
    determinant = math.fsum(rows[0][i]*reciprocal[0][i] for i in range(3))
    if not math.isfinite(determinant) or determinant == 0:
        raise ValueError("Singular or nonfinite simulation lattice")
    result = [2*math.pi*math.fsum(twist[j]*reciprocal[j][i] for j in range(3))/determinant
              for i in range(3)]
    finite(result)
    return result


def parameter(parent, name):
    """Resolve exactly one ordinary XML parameter, rejecting spelling duplicates."""
    found = [item for item in parent.findall('parameter') if item.get('name', '').lower() == name.lower()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one parameter {name}")
    return found[0]


def positive_cap(value):
    """Keep explicit finite numeric admission; zero/unlimited budgets are not accepted."""
    cap = int(value)
    if cap <= 0 or cap >= 2**64-1:
        raise ValueError("Expected positive finite numeric byte cap")


def prepare(entry, base):
    """Validate one self-contained sole-component input and derive immutable run metadata."""
    if set(entry) != {'id', 'seed', 'fractional_twist', 'weight', 'input', 'measurement_series', 'discard_blocks'}:
        raise ValueError("Unknown/missing twist fields; restart and prior-attempt adoption are not supported")
    identifier, seed = entry['id'], entry['seed']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or type(seed) is not int or not 0 < seed < 2**31:
        raise ValueError("Unsafe twist id or nonpositive/out-of-range integer seed")
    twist = [float(value) for value in entry['fractional_twist']]
    weight = float(entry['weight'])
    finite(twist + [weight])
    if len(twist) != 3 or weight < 0:
        raise ValueError("Expected three fractional components and nonnegative weight")
    source = (base / entry['input']).resolve()
    source_bytes = source.read_bytes()
    root = ET.fromstring(source_bytes)
    external = {'href', 'src', 'file', 'filename', 'fileroot', 'path'}
    if any(node.tag in ('include', 'loop') or external.intersection(key.lower() for key in node.attrib)
           for node in root.iter()):
        raise ValueError("First workflow cut requires self-contained XML without includes/assets/loops")
    has_restart = any(node.get('name', '').lower() == 'restart_path' for node in root.iter('parameter'))
    if has_restart or root.find('.//mcwalkerset') is not None:
        raise ValueError("Restart/walker import is not admitted by this first workflow cut")
    systems, waves = root.findall('qmcsystem'), root.findall('.//wavefunction')
    if (len(systems) != 1 or len(waves) != 1 or
            len(list(waves[0])) != 1 or waves[0][0].tag != 'psiformer'):
        raise ValueError("Expected one self-contained system and sole PsiFormer component")
    system, head = systems[0], waves[0][0]
    if {'parameters', 'configuration', 'export_parameters'}.intersection(key.lower() for key in head.attrib):
        raise ValueError("Only initialized self-contained heads without parameter/configuration/export paths are admitted")
    hamiltonians = system.findall('hamiltonian')
    if (len(hamiltonians) != 1 or len(list(hamiltonians[0])) != 3 or
            any(node.tag != 'pairpot' or node.get('type') != 'coulomb' or
                node.get('physical', 'yes').lower() not in ('yes', 'true', '1') for node in hamiltonians[0])):
        raise ValueError("Expected ordinary kinetic plus exactly three physical Coulomb pair potentials")
    target, source_name = waves[0].get('target'), head.get('source')
    if hamiltonians[0].get('target') != target:
        raise ValueError("Hamiltonian target must equal the wavefunction target")
    pairs = {(node.get('source'), node.get('target')) for node in hamiltonians[0]}
    if not target or not source_name or pairs != {(target, target), (source_name, source_name), (source_name, target)}:
        raise ValueError("Expected e/e, ion/ion and ion/e Coulomb identities")
    cell = system.find('simulationcell')
    if cell is None or parameter(cell, 'bconds').text.split() != ['p', 'p', 'p']:
        raise ValueError("Expected a fully periodic simulation cell")
    if cell.attrib or any(node.tag != 'parameter' or node.get('name') not in
                          ('lattice', 'bconds', 'LR_dim_cutoff') for node in cell):
        raise ValueError("Only explicit unscaled lattice/bconds/LR_dim_cutoff cell metadata is admitted")
    lattice = parameter(cell, 'lattice')
    if lattice.get('units', 'bohr').lower() != 'bohr':
        raise ValueError("Only explicit/default bohr lattice units are admitted")
    axes = [float(value) for value in lattice.text.split()]
    finite(axes)
    if len(axes) != 9:
        raise ValueError("Expected nine simulation-lattice components")
    kappa = cartesian_twist([axes[i:i+3] for i in (0, 3, 6)], twist)
    paired = head.get('initialization') == 'psiformer_complex_periodic_v1'
    if head.get('system') != 'all_electron' or head.get('feature_policy') != 'periodic_torus_v1':
        raise ValueError("Expected explicit periodic all-electron metadata")
    if head.get('optimize', 'no') != 'no' or head.get('precision', 'fp64_reference') != 'fp64_reference':
        raise ValueError("Only scalar optimize=no/fp64_reference is admitted")
    qmc = root.findall('qmc')
    methods = [item.get('method') for item in qmc]
    if methods not in (['vmc_batch'], ['wf_train', 'vmc_batch']):
        raise ValueError("Expected only fixed-P VMC, optionally after one CPU training stage")
    gpu = head.get('gpu', 'no').lower()
    if (gpu not in ('no', 'cpu', 'cuda', 'hip') or
            (len(qmc) == 2 and (not paired or gpu not in ('no', 'cpu')))):
        raise ValueError("Training requires paired CPU; GPU is initialized-only VMC, never restore")
    if paired:
        positive_cap(head.get('max_clone_numeric_bytes', '0'))
        if gpu in ('cuda', 'hip'):
            positive_cap(head.get('max_gpu_numeric_bytes', '0'))
    elif any(twist) or head.get('initialization') != 'deepqmc_psiformer_v1':
        raise ValueError("Optional real profile is initialized Gamma inference only")
    if len(qmc) == 2:
        training = qmc[0].find('training')
        if (training is None or training.get('objective') != 'energy' or
                training.get('optimizer') not in ('adam', 'sgd', 'sr')):
            raise ValueError("Expected a qualified CPU energy optimizer")
        positive_cap(parameter(training, 'maximum_retained_bytes').text)
        parameter(training, 'checkpoint_path').text = 'checkpoint.manifest.h5'
    series, discard = entry['measurement_series'], entry['discard_blocks']
    if type(series) is not int or series != len(qmc)-1 or type(discard) is not int or discard < 0:
        raise ValueError("Measurement must select the final fixed-P VMC series and nonnegative block discard")
    projects, randoms = root.findall('project'), root.findall('random')
    if len(projects) != 1 or len(randoms) != 1:
        raise ValueError("Explicit project/random input is required")
    project, random = projects[0], randoms[0]
    project.set('id', identifier)
    project.set('series', '0')
    random.set('seed', str(seed))
    head.set('twist', ' '.join(repr(value) for value in twist))
    # Exact fixed physical metadata includes spin populations, ions, masses and cell, not parameters.
    physical = b''.join(ET.tostring(node) for node in system if node.tag != 'wavefunction')
    return {'id': identifier, 'seed': seed, 'weight': weight, 'fractional_twist': twist,
            'cartesian_kappa_bohr_inverse': kappa, 'input': str(source),
            'input_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'initialization_profile': head.get('initialization'),
            'initialization_seed': head.get('initialization_seed', '0'),
            'measurement_series': series, 'discard_blocks': discard,
            'physical_sha256': hashlib.sha256(physical).hexdigest(), 'xml': root}


def load_statistics():
    """Reuse QMCPACK's existing autocorrelation-aware scalar utility, not a second estimator."""
    path = Path(__file__).resolve().parents[1] / 'tests/scripts/check_scalars.py'
    spec = importlib.util.spec_from_file_location('qmcpack_scalar_statistics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simstats


def measure(path, discard, simstats):
    """Read one explicitly selected fixed-parameter VMC series and retain its raw blocks."""
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith('#'):
        raise ValueError("Missing scalar header")
    columns = lines[0].split()[1:]
    index = columns.index('LocalEnergy')
    rows = [[float(value) for value in line.split()] for line in lines[1:]
            if line.strip() and not line.lstrip().startswith('#')]
    if any(len(row) != len(columns) for row in rows):
        raise ValueError("Malformed scalar extent")
    values = [row[index] for row in rows]
    finite(values)
    if len(values)-discard < 2:
        raise ValueError("Require at least two finite VMC blocks after discard (not a precision claim)")
    mean, variance, error, kappa = simstats(values, exclude=discard)
    finite([mean, variance, error, kappa])
    return {'mean_energy_hartree': mean, 'block_energy_variance': variance, 'mc_standard_error_hartree': error,
            'autocorrelation_factor': kappa, 'blocks': len(values)-discard, 'raw_block_energies': values,
            'scalar_file': str(path), 'scalar_sha256': sha256(path)}


def run_workflow(manifest_path, binary, output, timeout=300):
    """Launch fresh serial independent jobs; failed/partial attempts are never reused."""
    manifest_path, binary, output = manifest_path.resolve(), binary.resolve(), output.resolve()
    manifest = json.loads(manifest_path.read_text())
    if (set(manifest) != {'version', 'independent_streams', 'twists'} or
            manifest['version'] != 1 or manifest['independent_streams'] is not True):
        raise ValueError("Expected v1 explicitly independent twists; restart/reuse/covariance modes are not admitted")
    entries = [prepare(entry, manifest_path.parent) for entry in manifest['twists']]
    if not entries or any(
            len({tuple(entry[field]) if field == 'fractional_twist' else entry[field]
                 for entry in entries}) != len(entries) for field in ('id', 'seed', 'fractional_twist')):
        raise ValueError("Require nonempty unique ids/seeds/fractional representatives")
    if len({entry['physical_sha256'] for entry in entries}) != 1:
        raise ValueError("Twists must preserve the same fixed-N/spin/cell/ion/mass/Hamiltonian input")
    total = math.fsum(entry['weight'] for entry in entries)
    finite([total])
    if total <= 0:
        raise ValueError("Twist weight sum must be positive")
    simstats = load_statistics()
    binary_hash = sha256(binary)
    output.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix='attempt_', dir=output))
    (attempt / 'manifest.json').write_bytes(manifest_path.read_bytes())
    result = {'version': 1, 'status': 'incomplete', 'restart_supported': False,
              'independent_streams': True, 'manifest_sha256': sha256(manifest_path),
              'binary': str(binary), 'binary_sha256': binary_hash,
              'phase_convention': 'kappa=2*pi*A_inverse*t; XML lattice rows A; no fractional wrapping',
              'scope': 'raw fixed-N Ewald VMC energies; no grid-convergence/finite-size/precision claim', 'twists': []}
    for entry in entries:
        root = entry.pop('xml')
        directory = attempt / entry['id']
        directory.mkdir()
        original = Path(entry['input']).read_bytes()
        if hashlib.sha256(original).hexdigest() != entry['input_sha256']:
            raise ValueError("Source input changed after preparation")
        (directory / 'source.xml').write_bytes(original)
        xml = directory / 'input.xml'
        ET.ElementTree(root).write(xml, encoding='unicode', xml_declaration=True)
        entry.update(normalized_weight=entry['weight']/total, generated_input=str(xml),
                     generated_input_sha256=sha256(xml))
        # Record intent before launch: failure leaves raw XML/logs, never an apparently complete result.
        result['twists'].append(entry)
        (attempt / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
        # Stream console directly to disk: timeout/failure retains partial logs without unbounded PIPE storage.
        with (directory / 'qmcpack.log').open('w') as console:
            try:
                completed = subprocess.run([str(binary), xml.name], cwd=directory, stdout=console,
                                           stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"Twist {entry['id']} timed out; evidence {directory}") from error
        if completed.returncode:
            raise RuntimeError(f"Twist {entry['id']} failed ({completed.returncode}); evidence {directory}")
        scalar = directory / f"{entry['id']}.s{entry['measurement_series']:03d}.scalar.dat"
        entry.update(measure(scalar, entry['discard_blocks'], simstats))
        # Checkpoints are evidence only: the next slice must validate the entire distributed family.
        entry['checkpoint_artifact_sha256'] = {
            p.name: sha256(p) for p in sorted(directory.glob('checkpoint.manifest.h5*')) if p.is_file()}
    result.update(
        status='complete',
        mean_energy_hartree=math.fsum(e['normalized_weight']*e['mean_energy_hartree'] for e in entries),
        mc_standard_error_hartree=math.sqrt(math.fsum(
            (e['normalized_weight']*e['mc_standard_error_hartree'])**2 for e in entries)))
    (attempt / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return attempt / 'result.json'


def main():
    """Expose only fresh prepare/launch/aggregate; no implicit complete-attempt adoption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(run_workflow(args.manifest, args.binary, args.output))


if __name__ == '__main__':
    main()
