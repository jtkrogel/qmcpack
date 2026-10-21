#!/usr/bin/env python3
"""Tiny generated trace fixtures and the actual CPU ownership executable."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

READER = Path(__file__).resolve().parents[3] / 'utils' / 'psiformer_memory_report.py'
spec = importlib.util.spec_from_file_location('memory_report', READER)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
BINARY = sys.argv.pop(1) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else None


def event(kind, ident=1, **extra):
    result = dict(schema=1, kind=kind, mono_ns=2, thread=1, id=ident,
        owner='generated', category='device_capacity', backend='hip', device=0,
        bytes=0, logical_bytes=0, source='generated.cpp', line=1, parent=0,
        offset_bytes=0, dtype='bytes', shape='unknown', generation=0)
    result.update(extra)
    return result


class MemoryReportTests(unittest.TestCase):
    def report(self, events, finalize=True, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.jsonl'
            records = [dict(schema=1, kind='begin', wall_ns=1, mono_ns=1, pid=1, rank='unknown')]
            records += events
            if finalize:
                records += [dict(schema=1, kind='end', events=len(events))]
            path.write_text(''.join(json.dumps(e) + '\n' for e in records))
            return reader.summarize(path, **kwargs)

    def test_preexisting_nested_and_concurrent_phase_peaks(self):
        report = self.report([
            event('allocated', bytes=64), event('phase_begin', 2),
            event('phase_begin', 3, thread=2), event('allocated', 4, bytes=32),
            event('release', 4), event('phase_end', 3, thread=2),
            event('phase_end', 2), event('release'),
        ])
        self.assertTrue(report['complete'], report['warnings'])
        self.assertEqual(report['peak_bytes'], {'device_capacity:hip:0': 96})
        for phase in report['phases']:
            self.assertEqual(phase['entry_live_bytes']['device_capacity:hip:0'], 64)
            self.assertEqual(phase['peak_bytes']['device_capacity:hip:0'], 96)
        self.assertEqual(report['peak_owner_ids_in_trace_order']['device_capacity:hip:0'], [1, 4])

    def test_alias_does_not_charge_and_failed_free_remains_live(self):
        report = self.report([
            event('allocated', bytes=64), event('view', 2, parent=1, bytes=32, offset_bytes=16),
            event('free_failed', bytes=64),
        ])
        self.assertTrue(report['complete'])
        self.assertEqual(report['final_live_bytes']['device_capacity:hip:0'], 64)
        self.assertEqual(len(report['owners']), 1)
        self.assertEqual(len(report['packed_views']), 1)

    def test_capacity_replacement_not_double_charged(self):
        report = self.report([
            event('capacity', category='host_capacity', backend='host', device=-1, bytes=64),
            event('capacity', category='host_capacity', backend='host', device=-1, bytes=128),
            event('capacity', category='host_capacity', backend='host', device=-1, bytes=32),
            event('release', category='host_capacity', backend='host', device=-1),
        ])
        self.assertTrue(report['complete'])
        self.assertEqual(report['peak_bytes']['host_capacity:host:-1'], 128)
        self.assertEqual(report['final_live_bytes']['host_capacity:host:-1'], 0)

    def test_schema_extents_and_bounds(self):
        self.assertFalse(self.report([event('allocated', bytes=-1)])['complete'])
        self.assertFalse(self.report([event('allocated', schema=2)])['complete'])
        self.assertFalse(self.report([event('allocated', bytes=8), event('view', 2, parent=1, bytes=9)])['complete'])
        bound = self.report([event('allocated', bytes=8), event('allocated', 2, bytes=8)], max_metadata=1)
        self.assertTrue(any('bound' in warning for warning in bound['warnings']))
        truncated = self.report([event('rejected', 1), event('rejected', 2)], max_details=1)
        self.assertTrue(truncated['detail_truncated'])
        self.assertFalse(self.report([event('release')])['complete'])
        self.assertFalse(self.report([event('allocated', bytes=8)], finalize=False)['complete'])

    @unittest.skipUnless(BINARY, 'pass ownership test executable for integration checks')
    def test_real_ownership_stream_disabled_enabled_and_io_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'trace')
            environment = dict(os.environ)
            for name in ('PSIFORMER_MEMORY_TRACE', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'SLURM_PROCID'):
                environment.pop(name, None)
            off = subprocess.run([BINARY], env=environment, capture_output=True, text=True, check=True)
            self.assertIn('enabled=0', off.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])
            environment['PSIFORMER_MEMORY_TRACE'] = prefix
            subprocess.run([BINARY], env=environment, capture_output=True, text=True, check=True)
            trace = next(Path(directory).glob('*.jsonl'))
            report = reader.summarize(trace)
            self.assertTrue(report['complete'], report['warnings'])
            self.assertEqual(report['peak_bytes']['device_capacity:cuda:0'], 384)
            self.assertEqual(report['final_live_bytes']['device_capacity:cuda:0'], 128)
            self.assertEqual(report['native_counter_peak_bytes']['device_capacity:cuda:0'], 384)
            self.assertEqual(report['native_counter_last_live_bytes']['device_capacity:cuda:0'], 128)
            self.assertEqual(report['rejected_requests'][0]['owner'], 'test::rejected')
            self.assertEqual(report['rejected_requests'][0]['bytes'], 512)
            self.assertTrue(report['rejected_requests'][0]['source'].endswith('test_psiformer_memory_trace.cpp'))
            self.assertEqual(len(report['packed_views']), 1)
            self.assertEqual(len([p for p in report['phases'] if p['owner'] == 'concurrent']), 4)
            environment['PSIFORMER_MEMORY_TRACE'] = str(Path(directory) / 'destructor')
            environment['PSIFORMER_MEMORY_TEST_DESTRUCTOR_FAILURE'] = '1'
            subprocess.run([BINARY], env=environment, capture_output=True, text=True, check=True)
            destructor = reader.summarize(next(Path(directory).glob('destructor*.jsonl')))
            self.assertTrue(destructor['complete'])
            self.assertEqual(destructor['final_live_bytes']['device_capacity:cuda:0'], 192)
            del environment['PSIFORMER_MEMORY_TEST_DESTRUCTOR_FAILURE']
            environment['PSIFORMER_MEMORY_TRACE'] = str(Path(directory) / 'missing' / 'trace')
            invalid = subprocess.run([BINARY], env=environment, capture_output=True, text=True, check=True)
            self.assertEqual(invalid.stderr.count('memory trace I/O failure'), 1)
            environment['PSIFORMER_MEMORY_TRACE'] = str(Path(directory) / 'full')
            environment['PSIFORMER_MEMORY_TEST_IO_FAILURE'] = '1'
            full = subprocess.run([BINARY], env=environment, capture_output=True, text=True, check=True)
            self.assertEqual(full.stderr.count('memory trace I/O failure'), 1)
            del environment['PSIFORMER_MEMORY_TEST_IO_FAILURE']
            environment['PSIFORMER_MEMORY_TEST_LATE'] = '1'
            environment['PSIFORMER_MEMORY_TRACE'] = str(Path(directory) / 'late')
            subprocess.run([BINARY], env=environment, capture_output=True, text=True, check=True)
            late = reader.summarize(next(Path(directory).glob('late*.jsonl')))
            self.assertFalse(late['complete'])
            self.assertTrue(any('late static' in warning for warning in late['warnings']))


if __name__ == '__main__':
    unittest.main()
