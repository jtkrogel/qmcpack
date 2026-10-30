#!/usr/bin/env python3
"""Focused schema and aggregation tests for optional WFTrain work records."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

READER = Path(__file__).resolve().parents[3] / 'utils' / 'psiformer_work_report.py'
spec = importlib.util.spec_from_file_location('work_report', READER)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def row(rank, **extra):
    result = dict(schema=1, kind='work_counters', qmc_section='training', phase='iteration', iteration=4,
        rank=str(rank), pid=100 + int(rank), mono_ns=20, **{field: 0 for field in reader.COUNTERS})
    result.update(extra)
    return result


class WorkReportTests(unittest.TestCase):
    def report(self, records):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, rows in enumerate(records):
                path = Path(directory) / f'rank{index}.jsonl'
                path.write_text(''.join(json.dumps(value) + '\n' for value in rows))
                paths.append(path)
            return reader.read_records(paths)

    def test_rank_records_sum_work_without_timing(self):
        first = row(0, proposal_rounds=4, requested_value_product_batches=5,
                    computed_product_status_unknown_batches=1, hamiltonian_walker_evaluations=8)
        second = row(1, proposal_rounds=3, requested_value_product_batches=6,
                     computed_value_product_batches=4, hamiltonian_walker_evaluations=10)
        report = self.report([[first], [second]])
        self.assertTrue(report['complete'], report['warnings'])
        self.assertEqual(len(report['records']), 2)
        total = report['totals'][0]
        self.assertEqual(total['proposal_rounds'], 7)
        self.assertEqual(total['requested_value_product_batches'], 11)
        self.assertEqual(total['computed_product_status_unknown_batches'], 1)
        self.assertEqual(total['hamiltonian_walker_evaluations'], 18)
        self.assertIn('not elapsed time', report['scope'])

    def test_malformed_negative_duplicate_and_missing_inputs_are_reported(self):
        self.assertFalse(self.report([[row(0, accepted_move_decisions=-1)]])['complete'])
        duplicate = row(0)
        self.assertFalse(self.report([[duplicate, duplicate]])['complete'])
        with tempfile.TemporaryDirectory() as directory:
            report = reader.read_records([Path(directory) / 'missing.jsonl'])
        self.assertFalse(report['complete'])

    def test_qmc_section_disambiguates_same_iteration_records(self):
        first = row(0, qmc_section='pre"train\\phase', proposal_rounds=1)
        second = row(0, qmc_section='inference', proposal_rounds=2)
        report = self.report([[first, second]])
        self.assertTrue(report['complete'], report['warnings'])
        self.assertEqual(len(report['totals']), 2)
        self.assertEqual({item['qmc_section'] for item in report['totals']},
                         {'pre"train\\phase', 'inference'})
        self.assertEqual(sum(item['proposal_rounds'] for item in report['totals']), 3)

        legacy = row(1)
        legacy.pop('qmc_section')
        report = self.report([[legacy]])
        self.assertTrue(report['complete'], report['warnings'])
        self.assertEqual(report['records'][0]['qmc_section'], 'unknown')


if __name__ == '__main__':
    unittest.main()
