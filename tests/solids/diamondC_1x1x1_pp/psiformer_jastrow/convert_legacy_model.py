#!/usr/bin/env python3
"""Explicitly convert a real Gamma bare-neural native model to the zero-residual pair policy."""
import argparse
import hashlib
import math
from pathlib import Path
import shutil
import struct
import sys

import h5py
import numpy as np

MASK = (1 << 64)-1


def scalar_text(dataset):
    value = dataset[...].reshape(-1)[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def fingerprint(data, seed):
    value = seed
    for byte in data:
        value = ((value ^ byte)*1099511628211) & MASK
    return f'{value:016x}'


def layout_fingerprint(modules, names, ranks, shapes, offsets, count):
    data = bytearray(struct.pack('<QQ',len(modules),count))
    for i,(module,name) in enumerate(zip(modules,names)):
        for text in (module,name):
            encoded = text.encode()
            data.extend(struct.pack('<Q',len(encoded)))
            data.extend(encoded)
        rank = int(ranks[i])
        data.extend(struct.pack('<Q',rank))
        for extent in shapes[i,:rank]:
            data.extend(struct.pack('<Q',int(extent)))
        data.extend(struct.pack('<QQ',int(offsets[i]),int(offsets[i+1])))
    return fingerprint(data,14695981039346656037)


def put_string(group,name,value):
    if name in group:
        del group[name]
    group.create_dataset(name,data=np.array([value.encode()],dtype=f'S{len(value)}'))


def convert(source,destination,cutoff,same_reference,opposite_reference):
    if sys.byteorder != 'little':
        raise ValueError('This bounded native conversion currently requires a little-endian CPU host.')
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError('Conversion requires a new destination; the source is immutable.')
    for value in (cutoff,same_reference,opposite_reference):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Cutoff and fixed reference lengths must be finite and positive.')
    with h5py.File(source,'r') as model:
        metadata = model['psiformer_model']
        if (metadata['format_version'][...].tolist() != [1,0,0] or
            int(metadata['paired_complex'][()]) != 0 or int(metadata['periodic'][()]) != 1 or
            scalar_text(metadata['feature_policy']) != 'periodic_torus_v1' or
            scalar_text(metadata['system']) != 'pseudopotential' or
            scalar_text(metadata['precision']) != 'fp64_reference' or
            np.any(metadata['twist'][...] != 0) or 'jastrow_neural_product' in model):
            raise ValueError('Expected a real FP64 Gamma periodic pseudopotential bare-neural standalone model.')
        values = model['values'][...]
        if values.ndim != 1 or values.dtype.itemsize != 8 or not np.all(np.isfinite(values)):
            raise ValueError('Invalid native parameter values.')
        expected_values_hash = fingerprint(struct.pack('<Q',values.size)+values.astype('<f8').tobytes(),1469598103934665603)
        if scalar_text(metadata['values_fingerprint']) != expected_values_hash:
            raise ValueError('Source parameter fingerprint mismatch.')
        modules = [x.decode() for x in model['layout/modules'][...]]
        names = [x.decode() for x in model['layout/names'][...]]
        ranks,shapes,offsets = (model[f'layout/{name}'][...] for name in ('ranks','shapes','offsets'))
        if scalar_text(metadata['layout_fingerprint']) != layout_fingerprint(modules,names,ranks,shapes,offsets,values.size):
            raise ValueError('Source layout fingerprint mismatch.')
        lattice = metadata['lattice'][...].reshape(3,3)
        if not np.all(np.isfinite(lattice)):
            raise ValueError('Nonfinite source lattice.')
        reciprocal = np.linalg.inv(lattice).T
        unique_radius = float(np.min(.5/np.linalg.norm(reciprocal,axis=1)))
        if cutoff >= unique_radius:
            raise ValueError(f'Cutoff must lie strictly inside the reciprocal-face sphere ({unique_radius}).')
        rows = []
        channels = set()
        for i,(module,name) in enumerate(zip(modules,names)):
            tensor = values[int(offsets[i]):int(offsets[i+1])].copy()
            if module.endswith('/electronic_cusp_asymptotic'):
                if int(ranks[i]) != 0 or tensor.size != 1 or name not in ('same_alpha','anti_alpha'):
                    raise ValueError('Unsupported legacy cusp tensor.')
                channels.add(name)
                module = module.rsplit('/',1)[0]+'/residual_pair_v1'
                name = 'same_log_length' if name == 'same_alpha' else 'opposite_log_length'
                tensor[:] = 0
            rows.append((module,name,int(ranks[i]),shapes[i].copy(),tensor))
        populations = metadata['shape'][...][:2]
        expected_channels = {'anti_alpha'} | ({'same_alpha'} if max(populations)>1 else set())
        if channels != expected_channels:
            raise ValueError('Missing legacy pair-length tensors or already-converted source.')
    rows.sort(key=lambda row: (row[0],row[1]))
    offsets = np.zeros(len(rows)+1,dtype=np.int64)
    for i,row in enumerate(rows):
        offsets[i+1] = offsets[i]+row[4].size
    values = np.concatenate([row[4] for row in rows])
    modules,names = [row[0] for row in rows],[row[1] for row in rows]
    ranks,shapes = np.array([row[2] for row in rows],dtype=np.int64),np.stack([row[3] for row in rows])
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    temporary = destination.with_name(destination.name+'.conversion.tmp')
    if temporary.exists():
        raise ValueError('Conversion temporary destination already exists.')
    try:
        shutil.copy2(source,temporary)
        with h5py.File(temporary,'r+') as model:
            model['values'][...] = values
            for name,data in (('modules',modules),('names',names)):
                del model[f'layout/{name}']
                model['layout'].create_dataset(name,data=data,dtype=h5py.string_dtype('ascii'))
            for name,data in (('ranks',ranks),('shapes',shapes),('offsets',offsets)):
                model[f'layout/{name}'][...] = data
            metadata = model['psiformer_model']
            metadata['format_version'][...] = [1,3,0]
            metadata['parameter_version'][...] = 0
            put_string(metadata,'profile','psiformer_periodic_jastrow_v1')
            put_string(metadata,'layout_fingerprint',layout_fingerprint(modules,names,ranks,shapes,offsets,values.size))
            put_string(metadata,'values_fingerprint',fingerprint(struct.pack('<Q',values.size)+values.astype('<f8').tobytes(),1469598103934665603))
            residual = metadata.create_group('residual_pair')
            residual.create_dataset('version',data=np.uint32(1))
            residual.create_dataset('same_reference',data=np.float64(same_reference))
            residual.create_dataset('opposite_reference',data=np.float64(opposite_reference))
            residual.create_dataset('cutoff',data=np.float64(cutoff))
            put_string(residual,'policy','zero_residual_pair_v1')
            put_string(residual,'conversion_source_sha256',source_sha)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'Converted {source} -> {destination}; source_sha256={source_sha}; lambda=exact_zero; optimizer/sampler=fresh.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--cutoff',type=float,required=True)
    parser.add_argument('--same-reference',type=float,default=1)
    parser.add_argument('--opposite-reference',type=float,default=1)
    args = parser.parse_args()
    convert(args.source,args.destination,args.cutoff,args.same_reference,args.opposite_reference)


if __name__ == '__main__':
    main()
