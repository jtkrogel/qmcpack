#! /usr/bin/env python3

"""Report and plot standard and mean-field-residual energies."""

import argparse
import os
import sys

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
nexus_root = os.path.normpath(os.path.join(script_dir, '..', 'nexus'))
sys.path.insert(0,nexus_root)

from nexus.numerics import simstats


def read_scalar_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as scalar_file:
        quantities = scalar_file.readline().split()[2:]

    samples = np.loadtxt(filepath)
    samples = np.atleast_2d(samples)
    if samples.shape[1] != len(quantities)+1:
        raise ValueError('scalar.dat header does not match its data columns')

    block_index = samples[:,0]
    data = {name:samples[:,index+1] for index,name in enumerate(quantities)}
    return block_index,data


def report_statistics(name,samples):
    mean,variance,error,kappa = simstats(samples)
    print(f'{name:24s} = {mean: .12f} +/- {error:.12f}')


def trace_plot(filepath,block_index,traces,ylabel):
    figure,axis = plt.subplots(figsize=(9,5))
    for name,values in traces:
        axis.plot(block_index,values,label=name,linewidth=1.0)
    axis.set_xlabel('Block')
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    #figure.savefig(filepath,dpi=160)
    #plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description='Report and plot MFR energies from a no-Jastrow scalar.dat file.'
        )
    parser.add_argument('scalar_file',help='QMCPACK scalar.dat file')
    parser.add_argument('-e','--equilibration',type=int,default=0,
                        help='number of initial blocks to discard (default: 0)')
    parser.add_argument('--no-plots',action='store_true',
                        help='report statistics without producing trace plots')
    arguments = parser.parse_args()

    block_index,data = read_scalar_file(arguments.scalar_file)
    required = (
        'LocalEnergy',
        'TotalEnergyMF',
        'KineticJastrowResidual',
        'ElecElec',
        'MFRPotential',
        )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError('missing required scalar.dat columns: '+', '.join(missing))

    equilibration = arguments.equilibration
    sample_count = len(data[required[0]])
    if equilibration < 0 or equilibration >= sample_count:
        raise ValueError('equilibration must be nonnegative and smaller than the block count')

    if not np.allclose(data['KineticJastrowResidual'],0.0,rtol=0.0,atol=1e-12):
        raise ValueError('TotalEnergyMF_stat requires a no-Jastrow scalar.dat file')
    if not np.all(data['TotalEnergyMF'] == data['TotalEnergyMF'][0]):
        raise ValueError('TotalEnergyMF is not constant in scalar.dat')

    local_energy = data['LocalEnergy']
    local_energy_mfr = (data['TotalEnergyMF'] +
                         data['KineticJastrowResidual'] +
                         data['ElecElec'] - data['MFRPotential'])
    total_energy_mf_stat = (data['LocalEnergy'] - data['ElecElec'] +
                            data['MFRPotential'])
    total_energy_mf = data["TotalEnergyMF"]
    selected = slice(equilibration,None)

    report_statistics('LocalEnergy',local_energy[selected])
    report_statistics('LocalEnergyMFR',local_energy_mfr[selected])
    report_statistics('TotalEnergyMF_stat',total_energy_mf_stat[selected])
    print(f"{'TotalEnergyMF':24s} = {total_energy_mf[0]: .12f} (constant)")

    if not arguments.no_plots:
        suffix = '.scalar.dat'
        if arguments.scalar_file.endswith(suffix):
            plot_prefix = arguments.scalar_file[:-len(suffix)]
        else:
            plot_prefix = arguments.scalar_file
        trace_plot(
            plot_prefix+'.local_energy_trace.png',
            block_index[selected],
            [
                ('LocalEnergy',local_energy[selected]),
                ('LocalEnergyMFR',local_energy_mfr[selected]),
                ],
            'Energy (Ha)',
            )
        trace_plot(
            plot_prefix+'.total_energy_mf_trace.png',
            block_index[selected],
            [
                ('TotalEnergyMF_stat',total_energy_mf_stat[selected]),
                ('TotalEnergyMF',total_energy_mf[selected]),
                ],
            'Energy (Ha)',
            )


if __name__ == '__main__':
    main()

    plt.show()
