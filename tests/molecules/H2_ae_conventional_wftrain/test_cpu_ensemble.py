"""Cheap local checks; these never launch MPI or evaluate a model."""
import json
from pathlib import Path
import tempfile
import unittest

import check_cpu_ensemble as gate


class EnsemblePreparationTests(unittest.TestCase):
    def test_corresponding_slots_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            gate.prepare(root, Path(__file__).resolve().parents[3], Path("/unused/app"), "0" * 64)
            manifest = json.loads((root / "manifest.json").read_text())
            hashes = manifest["inputs"]
            self.assertEqual(hashes["mixed/slot0.xml"], hashes["sj_pair/slot0.xml"])
            self.assertEqual(hashes["mixed/slot1.xml"], hashes["mdj_pair/slot1.xml"])
            self.assertNotEqual(hashes["mixed/slot0.xml"], hashes["mixed/slot1.xml"])
            self.assertEqual(manifest["per_launch_seconds"], 35)
            self.assertEqual(manifest["aggregate_seconds"], 120)
            self.assertFalse((root / "execution.json").exists())
            with self.assertRaises(FileExistsError):
                gate.prepare(root, Path(__file__).resolve().parents[3], Path("/unused/app"), "0" * 64)

    def test_complete_finite_inference_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scalar.dat"
            header = "# index LocalEnergy BlockWeight BlockCPU AcceptRatio\n"
            path.write_text(header + "0 -1 4 0.01 1\n# complete yes\n")
            self.assertNotIn("BlockCPU", gate.scalar_rows(path))
            for row in ("0 nan 4 0.01 1", "0 -1 8 0.01 1", "0 -1 4 0.01 0"):
                path.write_text(header + row + "\n# complete yes\n")
                with self.assertRaises(AssertionError):
                    gate.scalar_rows(path)
            path.write_text(header + "0 -1 4 0.01 1\n")
            with self.assertRaises(AssertionError):
                gate.scalar_rows(path)


if __name__ == "__main__":
    unittest.main()
