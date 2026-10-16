#!/usr/bin/env python3
"""Supplemental estimator requests must leave training science/output unchanged."""
import argparse
import csv
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('baseline', type=Path)
parser.add_argument('requested', type=Path)
parser.add_argument('--h5diff', required=True)
args = parser.parse_args()
for root in (args.baseline, args.requested):
    assert not Path(str(root) + '.scalar.dat').exists()
    assert not Path(str(root) + '.stat.h5').exists()
    directory = Path(str(root) + '.training')
    assert not list(directory.rglob('*.stat.h5'))
    assert [p.name for p in directory.glob('*.scalar.dat')] == [root.name + '.i0000000.scalar.dat']
for suffix in ('state.h5', 'rank_000000.h5'):
    paths = []
    for root in (args.baseline, args.requested):
        iteration = root.name + '.i0000000'
        paths.append(Path(str(root) + '.training') / (iteration + '.checkpoint') /
                     (iteration + '.' + suffix))
    subprocess.run([args.h5diff, *map(str, paths)], check=True)
histories = []
for root in (args.baseline, args.requested):
    with Path(str(root) + '.training_info.csv').open() as stream:
        histories.append(list(csv.DictReader(stream)))
for field in ('iteration', 'sample_version', 'parameter_version', 'samples',
              'mean_energy', 'variance', 'gradient_norm', 'selected_alpha', 'committed'):
    assert [row[field] for row in histories[0]] == [row[field] for row in histories[1]], field
print('Supplemental scalar/operator requests preserve model, optimizer, sampler and scalar-only output')
