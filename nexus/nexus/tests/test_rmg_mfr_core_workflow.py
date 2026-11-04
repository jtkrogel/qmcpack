from pathlib import Path

from . import isolate_nexus_core


@isolate_nexus_core
def test_rmg_core_action_flags_and_results(tmp_path):
    from ..machines import job
    from ..rmg import generate_rmg
    from ..rmg_input import RmgInput, input_spec

    assert input_spec.keywords.write_mfr_core_action.default is False
    assert input_spec.keywords.write_mfr_core_diagnostics.default is False

    sim = generate_rmg(
        job=job(machine='ws1', cores=1),
        output_wave_function_file='Waves/core',
        write_mfr_core_action=True,
        write_mfr_core_diagnostics=True,
    )
    sim.locdir = str(tmp_path)

    assert sim.check_result('mfr_core_action', None)
    assert sim.check_result('orbitals', None)
    assert not sim.check_result('mfr_potential', None)
    assert not sim.check_result('unknown', None)
    assert sim.get_result('orbitals', None).h5file == str(tmp_path / 'Waves/core.h5')
    assert sim.get_result('mfr_core_action', None).h5file == str(tmp_path / 'Waves/core.mfr_core.h5')
    assert sim.get_output_files() == ['Waves/core.mfr_core.h5', 'Waves/core.h5']

    restored = RmgInput()
    restored.read_text(sim.input.write_text())
    assert restored.write_mfr_core_action is True
    assert restored.write_mfr_core_diagnostics is True


@isolate_nexus_core
def test_rmg_core_action_completion_requires_writer_marker_and_file(tmp_path):
    from ..machines import job
    from ..rmg import generate_rmg

    sim = generate_rmg(
        job=job(machine='ws1', cores=1),
        output_wave_function_file='scf',
        write_mfr_core_action=True,
    )
    sim.locdir = str(tmp_path)
    sim.job.finished = True
    log = tmp_path / (sim.infile + '.00.log')
    base_output = (
        'potential convergence has been achieved. stopping ...\n'
        'TIMING INFORMATION FOR THREADS\n'
    )
    orbitals = Path(sim.get_result('orbitals', None).h5file)
    core_action = Path(sim.get_result('mfr_core_action', None).h5file)
    orbitals.write_bytes(b'orbitals')

    log.write_text(base_output)
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded

    log.write_text(base_output + 'Wrote converged MFR core actions to scf.mfr_core.h5\n')
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded  # A marker without the file is incomplete.

    core_action.write_bytes(b'core-action')
    sim.check_sim_status()
    assert sim.succeeded and not sim.failed

    retry_log = tmp_path / (sim.infile + '.01.log')
    retry_log.write_text(base_output)
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded  # The latest attempt needs its own marker.


@isolate_nexus_core
def test_rmg_scalar_mfr_route_remains_unchanged(tmp_path):
    from ..machines import job
    from ..rmg import generate_rmg

    sim = generate_rmg(
        job=job(machine='ws1', cores=1),
        output_wave_function_file='scf',
        write_mfr_potential=True,
    )
    sim.locdir = str(tmp_path)

    assert sim.check_result('mfr_potential', None)
    assert sim.check_result('orbitals', None)
    assert not sim.check_result('mfr_core_action', None)
    assert sim.get_result('mfr_potential', None).h5file == str(tmp_path / 'scf.mfr.h5')
    assert sim.get_output_files() == ['scf.mfr.h5', 'scf.h5']

    sim.job.finished = True
    log = tmp_path / (sim.infile + '.00.log')
    log.write_text(
        'potential convergence has been achieved. stopping ...\n'
        'Wrote converged MFR potential to scf.mfr.h5\n'
        'TIMING INFORMATION FOR THREADS\n'
    )
    Path(sim.get_result('orbitals', None).h5file).write_bytes(b'orbitals')
    Path(sim.get_result('mfr_potential', None).h5file).write_bytes(b'potential')
    sim.check_sim_status()
    assert sim.succeeded and not sim.failed
