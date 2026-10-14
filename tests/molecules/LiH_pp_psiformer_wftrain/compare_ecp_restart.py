#!/usr/bin/env python3
"""Compare uninterrupted/restarted ECP training state exactly, not just energy."""
import argparse
import csv
from pathlib import Path
import re
import subprocess


def manifest_string(manifest, dataset, h5dump):
    """Read one manifest string using HDF5 tools, without a Python HDF5 dependency."""
    output = subprocess.check_output(
        [h5dump, "-d", "/wftrain_distributed_manifest/" + dataset, str(manifest)], text=True)
    return re.findall(r'"([^"]+)"', output)[-1]


def envelopes(manifest, h5dump):
    """Resolve the committed root envelope and rank-zero sampler shard."""
    root = manifest_string(manifest, "root_envelope_filename", h5dump)
    prefix = manifest_string(manifest, "sampler_shard_prefix", h5dump)
    return manifest.parent / root, manifest.parent / (prefix + ".rank-00000000.h5")


def main():
    """Require bitwise-equivalent saved optimizer, parameter, sampler and grid data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continuous", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("--h5diff", required=True)
    parser.add_argument("--h5dump", required=True)
    args = parser.parse_args()
    for original, resumed in zip(envelopes(args.continuous, args.h5dump),
                                 envelopes(args.resumed, args.h5dump)):
        subprocess.run([args.h5diff, str(original), str(resumed)], check=True)
    histories = []
    for manifest in [args.continuous, args.resumed]:
        name = "lih_pp_ecp_continuous" if manifest == args.continuous else "lih_pp_ecp_restart"
        with (manifest.parent / (name + ".s000.training.csv")).open() as stream:
            histories.append(list(csv.DictReader(stream)))
    original, resumed = histories[0][-1], histories[1][-1]
    for field in ["iteration", "sample_version", "parameter_version", "samples", "mean_energy",
                  "variance", "gradient_norm", "selected_alpha", "committed"]:
        if original[field] != resumed[field]:
            raise AssertionError("Restarted iteration history mismatch: " + field)
    print("ECP restart matches uninterrupted parameters, optimizer, walkers, RNGs and grids exactly")


if __name__ == "__main__":
    main()
