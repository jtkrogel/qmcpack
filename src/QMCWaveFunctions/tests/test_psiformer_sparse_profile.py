#!/usr/bin/env python3
"""Cheap controller contract tests; never invoke Slurm or a numerical model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_psiformer_sparse_profile as profile


class SparseProfileContracts(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "caps": {"case_seconds": 90, "campaign_seconds": 900, "maximum_attempts": 10,
                     "automatic_retry": False, "aggregate_cgroup_bytes": 14 * 1024**3,
                     "stop_observed_bytes": int(0.8 * 14 * 1024**3)},
            "cases": [{"id": "n50_full_w1", "request": "complete_vgl", "walkers": 1}]
        }

    def test_valid_inert_manifest(self):
        profile.validate_manifest(self.manifest)

    def test_exit_race_is_recoverable(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(profile.os, "killpg", side_effect=ProcessLookupError):
            profile.terminate(process)
        process.wait.assert_called_once_with()

    def test_no_oversized_or_ambiguous_campaign(self):
        for key, bad in (("case_seconds", 91), ("campaign_seconds", 901),
                         ("maximum_attempts", 11), ("aggregate_cgroup_bytes", 15 * 1024**3),
                         ("automatic_retry", True)):
            manifest = copy.deepcopy(self.manifest)
            manifest["caps"][key] = bad
            with self.assertRaises(AssertionError):
                profile.validate_manifest(manifest)
        self.manifest["cases"].append(self.manifest["cases"][0])
        with self.assertRaises(AssertionError):
            profile.validate_manifest(self.manifest)

    def test_unsupported_numerical_axes(self):
        for key, bad in (("walkers", 8), ("target_tile", 8), ("configuration_tile", 4),
                         ("request", "active_gradient"), ("ranks", 2)):
            manifest = copy.deepcopy(self.manifest)
            manifest["cases"][0][key] = bad
            with self.assertRaises(AssertionError):
                profile.validate_manifest(manifest)

    def test_native_pilot_is_w1_only(self):
        self.manifest["cases"][0].update(request="native_vmc", walkers=1)
        profile.validate_manifest(self.manifest)
        self.manifest["cases"][0]["walkers"] = 2
        with self.assertRaises(AssertionError):
            profile.validate_manifest(self.manifest)

    def test_native_scalar_and_owner_contract(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "console.log").write_text("CPU target VGL: maxK=1,\nCPU target VGL crowd: live_clones=1,\n")
            scalar = root / "pilot.s000.scalar.dat"
            header = "# index LocalEnergy LocalEnergy_sq BlockWeight AcceptRatio\n"
            scalar.write_text(header + "0 -2 4 1 1\n")
            self.assertEqual(profile.validate_native_vmc(root)["LocalEnergy"], -2)
            for bad in ("0 nan 4 1 1\n", "0 -2 4 2 1\n", "0 -2 4 1 0\n"):
                scalar.write_text(header + bad)
                with self.assertRaises(AssertionError):
                    profile.validate_native_vmc(root)

    def test_all_results_compared_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            first, second = Path(name) / "first.json", Path(name) / "second.json"
            data = dict(configurations_completed=1, electrons_per_configuration=50,
                        values=[1., -3., 0., 4.])
            first.write_text(json.dumps(data))
            second.write_text(json.dumps(data))
            self.assertEqual(profile.compare_values(first, second)["compared_values"], 4)
            for bad in (5., float("nan"), float("inf")):
                data["values"][-1] = bad
                second.write_text(json.dumps(data))
                with self.assertRaises(AssertionError):
                    profile.compare_values(first, second)
            data["values"] = [1., -3., 0., 4.]
            data["electrons_per_configuration"] = 100
            second.write_text(json.dumps(data))
            with self.assertRaises(AssertionError):
                profile.compare_values(first, second)


if __name__ == "__main__":
    unittest.main()
