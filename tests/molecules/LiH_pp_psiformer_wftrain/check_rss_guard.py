#!/usr/bin/env python3
"""Require a valid full-input run to stop at its explicitly tiny RSS guard."""
import csv
from pathlib import Path
import subprocess
import sys

result = subprocess.run([sys.argv[1], "wftrain_rss_guard.xml"], text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
Path("rss_guard.console.log").write_text(result.stdout)
if result.returncode == 0 or "process RSS exceeds maximum_resident_bytes" not in result.stdout:
    raise SystemExit("Expected explicit measured-RSS rejection, not success or an unrelated failure")
if "(listing suppressed)" not in result.stdout:
    raise SystemExit("Targeted parameter-listing suppression was not exercised")
with Path("lih_pp_psiformer_wftrain_adam.s000.training_info.csv").open() as stream:
    if list(csv.DictReader(stream)):
        raise SystemExit("An optimization update ran after the startup RSS limit was exceeded")
print("Measured RSS guard rejects the run before optimization and preserves targeted startup reporting")
