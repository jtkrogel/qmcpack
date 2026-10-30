#!/usr/bin/env python3
"""Validate and aggregate optional rank-local WFTrain work-counter JSONL."""
import argparse
import collections
import json
import sys


COUNTERS = (
    'proposal_rounds', 'ratio_api_batches', 'drift_gradient_api_batches',
    'accepted_move_decisions', 'rejected_move_decisions',
    'requested_value_product_batches', 'requested_gradient_product_batches',
    'requested_laplacian_product_batches', 'pre_move_full_spatial_handoff_batches',
    'computed_value_product_batches',
    'computed_gradient_product_batches', 'computed_laplacian_product_batches',
    'computed_product_status_unknown_batches', 'hamiltonian_walker_evaluations',
    'ecp_bearing_hamiltonian_walker_evaluations', 'retained_configurations',
    'kfac_factor_generation_builds', 'kfac_inverse_generation_builds',
)


def read_records(paths):
    records, warnings = [], []
    seen = set()
    for path in paths:
        last_mono_ns = -1
        try:
            with open(path, encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                        if row.get('schema') != 1 or row.get('kind') != 'work_counters':
                            raise ValueError('unsupported schema or record kind')
                        for field in ('phase', 'iteration', 'rank', 'pid', 'mono_ns', *COUNTERS):
                            if field not in row:
                                raise ValueError(f'missing field {field}')
                        row.setdefault('qmc_section', 'unknown')
                        if not isinstance(row['qmc_section'], str) or not row['qmc_section']:
                            raise ValueError('qmc_section must be a nonempty string')
                        if type(row['iteration']) is not int or row['iteration'] < -1:
                            raise ValueError('iteration must be -1 or nonnegative')
                        if not isinstance(row['phase'], str) or not row['phase']:
                            raise ValueError('phase must be a nonempty string')
                        for field in ('pid', 'mono_ns', *COUNTERS):
                            if type(row[field]) is not int or row[field] < 0:
                                raise ValueError(f'{field} must be a nonnegative integer')
                        if row['mono_ns'] < last_mono_ns:
                            raise ValueError('nonmonotonic record timestamp')
                        last_mono_ns = row['mono_ns']
                        key = (row['rank'], row['pid'], row['qmc_section'], row['phase'], row['iteration'])
                        if key in seen:
                            raise ValueError('duplicate rank/process/phase/iteration record')
                        seen.add(key)
                        records.append(dict(row, path=str(path), line=line_number))
                    except (ValueError, TypeError, KeyError) as error:
                        warnings.append(f'{path}:{line_number}: {error}')
                        break
        except OSError as error:
            warnings.append(f'{path}: {error}')

    totals = collections.defaultdict(lambda: {name: 0 for name in COUNTERS})
    for row in records:
        aggregate = totals[(row['qmc_section'], row['phase'], row['iteration'])]
        for name in COUNTERS:
            aggregate[name] += row[name]
    return dict(schema=1, complete=not warnings, warnings=warnings,
        scope='rank-local batch counts; sums across supplied ranks are work counts, not elapsed time',
        records=records, totals=[dict(qmc_section=qmc_section, phase=phase, iteration=iteration, **counts)
            for (qmc_section, phase, iteration), counts in sorted(totals.items())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', nargs='+')
    args = parser.parse_args()
    report = read_records(args.trace)
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0 if report['complete'] else 2


if __name__ == '__main__':
    sys.exit(main())
