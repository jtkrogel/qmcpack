#!/usr/bin/env python3
"""Parse lifecycle receipts emitted by the real CPU-only C++ receipt fixture."""
import json
import subprocess
import sys

output = subprocess.check_output([sys.argv[1]], text=True)
prefix = "PsiFormer capability "
lines = output.splitlines()
assert all(line.startswith(prefix) for line in lines)
receipts = [json.loads(line[len(prefix):]) for line in lines]
assert len(receipts) == 2
selection, summary = receipts
assert selection["schema"] == "cp0_v1"
assert selection["phase"] == "selection"
assert selection["requested_backend"] == "auto" and selection["selected_backend"] == "cpu"
assert selection["fallback_count"] == "1"
assert selection["active_crowds"] == "not_started"
assert selection["nominal_crowds"] == "driver_deferred"
assert selection["escaped"] == 'quote" backslash\\ newline\n tab\t'
assert "no_gpu_qualification" in selection["provenance"]
assert summary["phase"] == "dispatch_summary"
assert summary["model_identity"] == "42"
assert summary["scalar_full"] == "1" and summary["crowd_full"] == "400"
assert summary["maximum_crowd_walkers"] == "4"
assert summary["gpu_overlap"] == "not_measured"
