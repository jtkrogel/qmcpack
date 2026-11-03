#! /usr/bin/env python3

'''
Generate or run longer diamond MFR VMC calculations.

Set run to True to submit the calculations.  Run lengths and resources are
controlled by the local variables below.
'''

from nexus import settings,job,run_project
from nexus import generate_physical_system
from nexus import generate_qmcpack

run           = False
machine       = 'ws12'
qmcpack       = '/home/j1k/repositories/qmcpack_dev_new5/build_mfr_real/bin/qmcpack'
pseudo_dir    = '../tests/solids/diamondC_1x1x1_pp'
orbitals      = 'scf_mfr/pwscf_output/pwscf.pwscf.h5'
mfr_potential = 'scf_mfr/pwscf_output/pwscf.mfr.h5'
jastrow_file  = '../tests/solids/diamondC_1x1x1_pp/det_qmc_vmc_dmc_tm.in.xml'

cores       =  12
threads     =   1
walkers     =   cores
warmupsteps =  20
blocks      = 200
steps       =  10
substeps    =   3
timestep    = 0.3

settings(
    runs          = './jtk_mfr_runs/',
    results       = '',
    pseudo_dir    = pseudo_dir,
    generate_only = False,
    machine       = machine,
    sleep         = 1,
    )

system = generate_physical_system(
    units    = 'B',
    axes     = '''3.37316115  3.37316115  0.00000000
                  0.00000000  3.37316115  3.37316115
                  3.37316115  0.00000000  3.37316115''',
    elem_pos = '''
                 C  0.00000000  0.00000000  0.00000000
                 C  1.68658058  1.68658058  1.68658058
                 ''',
    kgrid    = (1,1,1),
    kshift   = (0,0,0),
    C        = 4,
    )

vmc_jastrow = generate_qmcpack(
    driver        = 'legacy',
    identifier    = 'mfr_vmc_jastrow',
    path          = 'vmc_jastrow_long',
    job           = job(cores=cores,threads=threads,app=qmcpack),
    input_type    = 'basic',
    system        = system,
    pseudos       = ['C.BFD.xml'],
    orbitals_h5   = orbitals,
    mfr_potential = mfr_potential,
    corrections   = [],
    seed          = 36,
    jastrows      = jastrow_file,
    qmc           = 'vmc',
    walkers       = walkers,
    warmupsteps   = warmupsteps,
    blocks        = blocks,
    steps         = steps,
    substeps      = substeps,
    timestep      = timestep,
    usedrift      = False,
    )

vmc_no_jastrow = generate_qmcpack(
    driver        = 'legacy',
    identifier    = 'mfr_vmc_no_jastrow',
    path          = 'vmc_no_jastrow_long',
    job           = job(cores=cores,threads=threads,app=qmcpack),
    input_type    = 'basic',
    system        = system,
    pseudos       = ['C.BFD.xml'],
    orbitals_h5   = orbitals,
    mfr_potential = mfr_potential,
    precision     = 'double',
    meshfactor    = 2.0,
    corrections   = [],
    seed          = 36,
    jastrows      = [],
    qmc           = 'vmc',
    walkers       = walkers,
    warmupsteps   = warmupsteps,
    blocks        = blocks,
    steps         = steps,
    substeps      = substeps,
    timestep      = timestep,
    usedrift      = False,
    )
#vmc_no_jastrow.show_input()


run_project()
