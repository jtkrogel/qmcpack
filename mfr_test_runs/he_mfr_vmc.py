#! /usr/bin/env python3

'''Run a Jastrow bcc-helium MFR VMC calculation.'''

from nexus import settings,job,run_project
from nexus import generate_physical_system
from nexus import generate_qmcpack

run           = True
machine       = 'ws12'
qmcpack       = '/home/j1k/repositories/qmcpack_dev_new5/build_mfr_real/bin/qmcpack'
pseudo_dir    = 'pseudos'
orbitals      = 'he_scf_mfr/pwscf_output/he.pwscf.h5'
mfr_potential = 'he_scf_mfr/pwscf_output/he.mfr.h5'

cores            =  12
threads          =   1
walkers_per_rank =  20
warmupsteps      =  20
blocks           = 800
steps            =  50
substeps         =   3
timestep         = 0.3

settings(
    runs          = 'he_mfr_runs',
    results       = '',
    pseudo_dir    = pseudo_dir,
    generate_only = not run,
    machine       = machine,
    sleep         = 1,
    )

system = generate_physical_system(
    units    = 'B',
    axes     = '''5.0  0.0  0.0
                  0.0  5.0  0.0
                  0.0  0.0  5.0''',
    elem_pos = '''
                 He  0.0  0.0  0.0
                 He  2.5  2.5  2.5
                 ''',
    kgrid    = (1,1,1),
    kshift   = (0,0,0),
    He       = 2,
    )

vmc_jastrow = generate_qmcpack(
    driver          = 'batched',
    identifier      = 'he_mfr_vmc_jastrow',
    path            = 'he_vmc_jastrow',
    job             = job(cores=cores,threads=threads,app=qmcpack),
    input_type      = 'basic',
    system          = system,
    pseudos         = ['He.ccECP.xml'],
    orbitals_h5     = orbitals,
    mfr_potential   = mfr_potential,
    mfr_qmc         = True,
    precision       = 'double',
    meshfactor      = 2.0,
    corrections     = [],
    seed            = 37,
    jastrows        = 'generateJ12',
    qmc             = 'vmc',
    walkers_per_rank = walkers_per_rank,
    warmupsteps     = warmupsteps,
    blocks          = blocks,
    steps           = steps,
    substeps        = substeps,
    timestep        = timestep,
    usedrift        = False,
    )

run_project()
