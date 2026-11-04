"""Dependency/XML wiring only; native file contents have separate integration tests."""
import os
import pytest

from . import isolate_nexus_core
from ..machines import job
from ..physical_system import generate_physical_system
from ..qmcpack import generate_qmcpack
from ..qmcpack_input import vmc
from ..rmg import generate_rmg


def helium_system():
    return generate_physical_system(
        axes=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        elem=['He', 'He'], pos=[[0, 0, 0], [2.5, 2.5, 2.5]],
        units='B', kgrid=(1, 1, 1), kshift=(0, 0, 0),
    )


@isolate_nexus_core
def test_rmg_core_action_qmcpack_dependency(tmp_path):
    standalone = False
    system = generate_physical_system(
        axes=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        elem=['He', 'He'], pos=[[0, 0, 0], [2.5, 2.5, 2.5]],
        units='B', kgrid=(1, 1, 1), kshift=(0, 0, 0),
    )
    scf = generate_rmg(
        job=job(machine='ws1', cores=1),
        write_mfr_core_action=True, output_wave_function_file='scf',
    )
    scf.locdir = str(tmp_path)
    for name in ('scf.h5', 'scf.mfr_core.h5'):
        (tmp_path / name).write_bytes(b'dependency fixture')
    qmc = generate_qmcpack(
        job=job(machine='ws1', cores=1), system=system, jastrows=[],
        mfr_core_action='dependency.mfr_core.h5', mfr_qmc=standalone,
        check_paths=False, calculations=[vmc(blocks=1, steps=1)],
    )
    qmc.locdir = str(tmp_path / 'qmc')
    qmc.incorporate_result('orbitals', scf.get_result('orbitals', qmc), scf)
    builder = qmc.input.simulation.qmcsystem.wavefunction.sposet_builders.bspline
    assert builder.href == os.path.join('..', 'scf.h5')
    action = scf.get_result('mfr_core_action', qmc)
    qmc.incorporate_result('mfr_core_action', action, scf)
    assert qmc.input.get('MFRCoreAction').href == os.path.join('..', 'scf.mfr_core.h5')
    assert qmc.input.simulation.calculations[0].mfr_qmc is standalone
    assert ('IonIon' in qmc.input.write_text()) is (not standalone)
    (tmp_path / 'scf.mfr_core.h5').unlink()
    with pytest.raises(FileNotFoundError):
        qmc.incorporate_result('mfr_core_action', action, scf)


@isolate_nexus_core
def test_rmg_core_dependency_requires_explicit_xml_mode(tmp_path):
    scf = generate_rmg(job=job(machine='ws1', cores=1), write_mfr_core_action=True)
    qmc = generate_qmcpack(job=job(machine='ws1', cores=1), system=helium_system())
    with pytest.raises(ValueError, match='requires mfr_core_action'):
        qmc.incorporate_result('mfr_core_action', None, scf)


@isolate_nexus_core
def test_rmg_standalone_core_dependency(tmp_path):
    scf = generate_rmg(job=job(machine='ws1', cores=1), write_mfr_core_action=True,
                       output_wave_function_file='scf')
    scf.locdir = str(tmp_path)
    (tmp_path / 'scf.mfr_core.h5').write_bytes(b'dependency fixture')
    qmc = generate_qmcpack(job=job(machine='ws1', cores=1), system=helium_system(),
                           mfr_core_action='dependency.mfr_core.h5', mfr_qmc=True,
                           check_paths=False, calculations=[vmc(blocks=1, steps=1)])
    qmc.locdir = str(tmp_path / 'qmc')
    qmc.incorporate_result('mfr_core_action', scf.get_result('mfr_core_action', qmc), scf)
    assert qmc.input.get('MFRCoreAction').href == os.path.join('..', 'scf.mfr_core.h5')
    assert qmc.input.simulation.calculations[0].mfr_qmc is True
    assert 'IonIon' not in qmc.input.write_text()
