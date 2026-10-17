"""Cheap preparation/admission checks: no MPI, model evaluation or network."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import check_complex_cpu_ensemble as gate


class ComplexEnsembleTests(unittest.TestCase):
    def test_matched_slots_and_inert_manifest(self):
        """Preserve WORLD-slot inputs and refuse execution before pin review."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            gate.prepare(root, "/unused/app", "0" * 64, "0" * 40, "/unused/lib")
            manifest = json.loads((root / "manifest.json").read_text())
            hashes = manifest["inputs"]
            self.assertEqual(hashes["mixed/slot0.xml"], hashes["a_pair/slot0.xml"])
            self.assertEqual(hashes["mixed/slot1.xml"], hashes["b_pair/slot1.xml"])
            self.assertEqual((manifest["threads_per_rank"], manifest["per_launch_seconds"], manifest["aggregate_seconds"]), (1, 35, 120))
            for label, slot, seed in (("mixed", 0, "271"), ("mixed", 1, "811")):
                tree = ET.parse(root / label / f"slot{slot}.xml")
                self.assertEqual(tree.find("random").get("seed"), seed)
                self.assertEqual(tree.find(".//psiformer").get("twist"), tree.find(".//sposet").get("twist"))
                self.assertEqual(tree.find(".//psiformer").get("optimize"), "no")
            with self.assertRaises(AssertionError):
                gate.execute(root, "unused")
            self.assertFalse((root / "execution.json").exists())

    def test_finite_complete_group_weight(self):
        """Reject world-sized sums, nonfinite output and incomplete inference."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scalar.dat"
            header = "# index LocalEnergy BlockWeight BlockCPU AcceptRatio\n"
            path.write_text(header + "0 -1 2 .01 1\n# complete yes\n")
            self.assertNotIn("BlockCPU", gate.scalar_rows(path)[1])
            for row in ("0 nan 2 .01 1", "0 -1 4 .01 1", "0 -1 2 .01 0"):
                path.write_text(header + row + "\n# complete yes\n")
                with self.assertRaises(AssertionError):
                    gate.scalar_rows(path)


if __name__ == "__main__":
    unittest.main()
