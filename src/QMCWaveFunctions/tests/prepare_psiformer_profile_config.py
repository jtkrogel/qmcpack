#!/usr/bin/env python3
"""Create a NEW fixed reference file from two existing accepted sampler shards.

This is HDF preparation only, not a model evaluation or sampler restart. Never
modify metadata, checkpoint shards or trained parameters in place.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


POSITIONS = "wftrain_sampler_shard/positions"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(metadata, shards, output):
    metadata, output = metadata.resolve(strict=True), output.resolve()
    shards = [path.resolve(strict=True) for path in shards]
    assert len(set(shards)) == 2, "two distinct rank shards are required"
    report_path = output.with_suffix(output.suffix + ".provenance.json")
    assert not output.exists() and not report_path.exists(), "refuse to overwrite preparation output"
    sources = {str(path): sha256(path) for path in [metadata] + shards}
    with h5py.File(metadata, "r") as source:
        assert int(source.attrs["n_up"]) == 10 and int(source.attrs["n_down"]) == 9
        assert int(source.attrs["n_determinants"]) == 16
        nuclei = np.asarray(source["nuclear_positions"][()])
        coordinates = []
        for shard in shards:
            with h5py.File(shard, "r") as handle:
                positions = np.asarray(handle[POSITIONS][()])
                assert positions.dtype == np.dtype("float64") and positions.shape == (2 * 19 * 3,)
                coordinates.append(positions.reshape(2, 19, 3))
        coordinates = np.concatenate(coordinates)
        assert coordinates.shape == (4, 19, 3) and np.isfinite(coordinates).all()
        differences = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        distances = np.linalg.norm(differences, axis=3)
        distances[:, np.arange(19), np.arange(19)] = np.inf
        minimum_ee = float(distances.min())
        minimum_ei = float(np.linalg.norm(coordinates[:, :, None, :] - nuclei[None, None, :, :], axis=3).min())
        assert np.isfinite(minimum_ee) and minimum_ee > 0
        assert np.isfinite(minimum_ei) and minimum_ei > 0
        with h5py.File(output, "x") as target:
            for name in source:
                if name != "electron_positions":
                    source.copy(name, target)
            for name, value in source.attrs.items():
                target.attrs[name] = value
            target.create_dataset("electron_positions", data=coordinates)
    assert sources == {str(path): sha256(path) for path in [metadata] + shards}, "source changed during preparation"
    # Prove everything except the explicitly replaced positions was retained.
    with h5py.File(metadata, "r") as source, h5py.File(output, "r") as target:
        assert source.keys() == target.keys() and source.attrs.keys() == target.attrs.keys()
        for name in source.attrs:
            np.testing.assert_array_equal(source.attrs[name], target.attrs[name])
        for name in source:
            if name != "electron_positions":
                np.testing.assert_array_equal(source[name][()], target[name][()])
    report = dict(kind="accepted NiH sampler reference coordinates; not a sampler restart",
                  source_sha256=sources, source_dataset=POSITIONS, shard_order=[str(path) for path in shards],
                  output=str(output), output_sha256=sha256(output), configurations=4,
                  electrons=19, spin_up=10, spin_down=9, determinants=16,
                  minimum_electron_distance=minimum_ee, minimum_electron_ion_distance=minimum_ei,
                  original_metadata_unchanged=True, nonposition_metadata_exact=True)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("rank0", type=Path)
    parser.add_argument("rank1", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.metadata, [args.rank0, args.rank1], args.output)


if __name__ == "__main__":
    main()
