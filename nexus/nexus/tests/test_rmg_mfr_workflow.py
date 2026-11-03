import os

import pytest

from . import isolate_nexus_core
from ..machines import job
from ..physical_system import generate_physical_system
from ..qmcpack import generate_qmcpack
from ..qmcpack_input import vmc
from ..rmg import generate_rmg
from ..rmg_input import RmgInput
from ..rmg_input import generate_rmg_input


@isolate_nexus_core
def test_rmg_mfr_results(tmp_path):
    sim = generate_rmg(
        job = job(machine='ws1',cores=1),
        write_mfr_potential = True,
        output_wave_function_file = 'Waves/scf',
        )
    sim.locdir = str(tmp_path)
    assert sim.check_result('orbitals',None)
    assert sim.check_result('mfr_potential',None)
    assert not sim.check_result('unknown',None)
    assert sim.get_result('orbitals',None).h5file == str(tmp_path/'Waves/scf.h5')
    assert sim.get_result('mfr_potential',None).h5file == str(tmp_path/'Waves/scf.mfr.h5')
    assert sim.get_output_files() == ['Waves/scf.mfr.h5','Waves/scf.h5']
    restored = RmgInput()
    restored.read_text(sim.input.write_text())
    assert restored.write_mfr_potential


@isolate_nexus_core
def test_rmg_mfr_completion(tmp_path):
    sim = generate_rmg(
        job = job(machine='ws1',cores=1),
        write_mfr_potential = True,
        output_wave_function_file = 'scf',
        )
    sim.locdir = str(tmp_path)
    sim.job.finished = True
    log = tmp_path/(sim.infile+'.00.log')
    log.write_text(
        'potential convergence has been achieved. stopping ...\n'
        'Wrote converged MFR potential to scf.mfr.h5\n'
        'TIMING INFORMATION FOR THREADS\n'
        )
    (tmp_path/'scf.h5').write_bytes(b'orbitals')
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded
    (tmp_path/'scf.mfr.h5').write_bytes(b'potential')
    sim.check_sim_status()
    assert sim.succeeded and not sim.failed
    sim.job.finished = False
    sim.check_sim_status()
    assert not sim.finished and not sim.succeeded and not sim.failed
    sim.job.finished = True
    retry = tmp_path/(sim.infile+'.01.log')
    retry.write_text('potential convergence has not been achieved\n')
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded
    retry.write_text('potential convergence has been achieved\n')
    sim.check_sim_status()
    assert sim.failed  # no completed writer or application termination


@isolate_nexus_core
def test_rmg_qmcpack_dependencies(tmp_path):
    system = generate_physical_system(
        axes = [[5,0,0],[0,5,0],[0,0,5]],
        elem = ['He','He'],
        pos = [[0,0,0],[2.5,2.5,2.5]],
        units = 'B',
        kgrid = (1,1,1),
        kshift = (0,0,0),
        )
    scf = generate_rmg(
        job = job(machine='ws1',cores=1),
        write_mfr_potential = True,
        output_wave_function_file = 'scf',
        )
    scf.locdir = str(tmp_path)
    for name in ('scf.h5','scf.mfr.h5'):
        (tmp_path/name).write_bytes(b'fixture')
    #end for
    qmc = generate_qmcpack(
        job = job(machine='ws1',cores=1),
        system = system,
        jastrows = [],
        mfr_potential = 'dependency.mfr.h5',
        mfr_qmc = False,
        check_paths = False,
        calculations = [vmc(blocks=1,steps=1)],
        )
    qmc.locdir = str(tmp_path/'qmc')
    qmc.incorporate_result('orbitals',scf.get_result('orbitals',qmc),scf)
    builder = qmc.input.simulation.qmcsystem.wavefunction.sposet_builders.bspline
    assert builder.href == os.path.join('..','scf.h5')
    qmc.incorporate_result('mfr_potential',scf.get_result('mfr_potential',qmc),scf)
    assert qmc.input.get('MFRPotential').href == os.path.join('..','scf.mfr.h5')
    assert qmc.input.simulation.calculations[0].mfr_qmc is False
    assert 'IonIon' in qmc.input.write_text()
    (tmp_path/'scf.mfr.h5').unlink()
    with pytest.raises(FileNotFoundError):
        qmc.incorporate_result('mfr_potential',scf.get_result('mfr_potential',qmc),scf)


@isolate_nexus_core
def test_rmg_explicit_twist():
    import numpy as np

    system = generate_physical_system(
        axes = [[5,0,0],[0,5,0],[0,0,5]],
        elem = ['He'],
        pos = [[0,0,0]],
        units = 'B',
        )
    system.structure.add_kpoints([[0.25,0.125,0]],kweights=[1.0],cell_unit=True)
    inp = generate_rmg_input(system=system,write_mfr_potential=True)
    assert np.array_equal(inp.kpoint_mesh,[0,0,0])
    assert np.allclose(inp.kpoints.kpoints,[[0.25,0.125,0]])
    assert inp.crds_units == 'Bohr'
    assert inp.is_valid()
