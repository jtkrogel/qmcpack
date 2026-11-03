import os
from pathlib import Path

import pytest

from . import NexusTestOrder
from . import create_pseudo_files, isolate_nexus_core

pytestmark = pytest.mark.order(NexusTestOrder.QMCPACK_CONVERTER_SIMULATIONS)


def test_pw2mfr_core_input_round_trip(tmp_path):
    from ..qmcpack_converters import Pw2mfrCoreInput
    from ..qmcpack_converters import generate_pw2mfr_core_input

    inp = generate_pw2mfr_core_input(
        prefix='carbon',
        outdir='pwscf_output',
        last_band=4,
        write_diagnostics=True,
        )

    written = tmp_path/'pw2mfr_core.in'
    inp.write(written)
    restored = Pw2mfrCoreInput(written)

    assert restored.inputpp.prefix == 'carbon'
    assert restored.inputpp.outdir == 'pwscf_output'
    assert restored.inputpp.last_band == 4
    assert restored.inputpp.write_diagnostics
#end def test_pw2mfr_core_input_round_trip


@isolate_nexus_core
def test_pw2mfr_core_to_qmcpack_dependency(tmp_path):
    from ..machines import job
    from ..nexus_base import nexus_config
    from ..qmcpack import generate_qmcpack
    from ..qmcpack_converters import Pw2mfrCore, generate_pw2mfr_core
    from ..qmcpack_input import vmc
    from .test_pwscf_simulation import get_pwscf_sim

    nexus_config.runs = ''
    nexus_config.local_directory = str(tmp_path)
    nexus_config.remote_directory = str(tmp_path)
    create_pseudo_files(tmp_path, ['C.BFD.upf', 'C.BFD.xml'])

    scf = get_pwscf_sim('scf')
    exporter = generate_pw2mfr_core(
        identifier='mfr_core',
        path='scf',
        job=job(machine='ws1', cores=1, app='pw2mfr_core.x'),
        prefix='pwscf',
        outdir='pwscf_output',
        dependencies=[(scf, 'orbitals')],
        )

    assert isinstance(exporter, Pw2mfrCore)
    assert exporter.check_result('mfr_core_action', None)
    assert not exporter.check_result('unknown', None)
    exporter.incorporate_result('orbitals', scf.get_result('orbitals', exporter), scf)

    action_result = exporter.get_result('mfr_core_action', None)
    action_path = Path(action_result.h5file)
    action_path.parent.mkdir(parents=True, exist_ok=True)
    action_path.touch()
    assert exporter.get_output_files() == [
        os.path.relpath(action_path, exporter.locdir)
        ]

    qmc = generate_qmcpack(
        identifier='qmc',
        path='qmc',
        job=job(machine='ws1', cores=1, app='qmcpack'),
        system=scf.system,
        pseudos=['C.BFD.xml'],
        jastrows=[],
        mfr_core_action='placeholder.mfr_core.h5',
        mfr_qmc=False,
        check_paths=False,
        calculations=[vmc(blocks=1, steps=1)],
        )
    qmc.incorporate_result('mfr_core_action', action_result, exporter)

    expected_href = os.path.relpath(action_path, qmc.locdir)
    assert qmc.input.get('MFRCoreAction').href == expected_href
    assert 'type="mfr_core_action"' in qmc.input.write_text()

#end def test_pw2mfr_core_to_qmcpack_dependency


@isolate_nexus_core
def test_pw2mfr_core_failed_rerun_with_previous_output(tmp_path):
    from ..machines import job
    from ..qmcpack_converters import generate_pw2mfr_core

    exporter = generate_pw2mfr_core(job=job(machine='ws1', cores=1))
    exporter.locdir = str(tmp_path)
    action = Path(exporter.get_result('mfr_core_action', None).h5file)
    action.parent.mkdir(parents=True, exist_ok=True)
    action.write_bytes(b'previous completed export')
    output = tmp_path/exporter.outfile
    exporter.job.finished = True

    output.write_text('Error: reference SCF is not converged\n')
    exporter.check_sim_status()
    assert exporter.failed and not exporter.finished and not exporter.succeeded
    output.write_text('Bare MFR core actions written to old.mfr_core.h5\n')
    exporter.check_sim_status()
    assert exporter.failed and not exporter.finished
    output.write_text('Bare MFR core actions written to current.mfr_core.h5\nJOB DONE.\n')
    exporter.check_sim_status()
    assert exporter.finished and exporter.succeeded and not exporter.failed
    exporter.job.finished = False
    exporter.check_sim_status()
    assert not exporter.finished and not exporter.succeeded and not exporter.failed
#end def test_pw2mfr_core_failed_rerun_with_previous_output
