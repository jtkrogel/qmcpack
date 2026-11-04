"""Prevent stale scalar/core results from releasing native RMG dependencies."""
from pathlib import Path

from . import isolate_nexus_core
from ..machines import job
from ..rmg import generate_rmg


def check_result_path_binding(tmp_path, kind, prefix):
    sim = generate_rmg(
        job = job(machine='ws1', cores=1),
        output_wave_function_file = 'Waves/scf',
        **{'write_mfr_'+kind: True},
        )
    sim.locdir = str(tmp_path)
    sim.job.finished = True
    result = sim.get_result('mfr_'+kind, None).h5file
    Path(result).parent.mkdir()
    Path(result).write_bytes(b'old result at expected path')
    Path(sim.get_result('orbitals', None).h5file).write_bytes(b'orbitals')
    marker = {'potential': 'Wrote converged MFR potential to ',
              'core_action': 'Wrote converged MFR core actions to '}[kind]
    base = ('potential convergence has been achieved. stopping ...\n'
            'TIMING INFORMATION FOR THREADS\n')
    log = tmp_path/(sim.infile+'.00.log')
    log.write_text(base+prefix+marker+'other/'+Path(result).name+'\n')
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded

    for current_path in ['Waves/./'+Path(result).name, result]:
        log.write_text(base+prefix+marker+current_path+'\n')
        sim.check_sim_status()
        assert sim.succeeded and not sim.failed
    #end for
    log.write_text(base+'not a writer message: '+marker+result+'\n')
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded
#end def check_result_path_binding


@isolate_nexus_core
def test_rmg_potential_plain_writer_binds_result_path(tmp_path):
    check_result_path_binding(tmp_path, 'potential', '')


@isolate_nexus_core
def test_rmg_potential_native_writer_binds_result_path(tmp_path):
    check_result_path_binding(tmp_path, 'potential', '[   0   4    4        6 ] common_prototypes1.h: ')


@isolate_nexus_core
def test_rmg_core_action_plain_writer_binds_result_path(tmp_path):
    check_result_path_binding(tmp_path, 'core_action', '')


@isolate_nexus_core
def test_rmg_core_action_native_writer_binds_result_path(tmp_path):
    check_result_path_binding(tmp_path, 'core_action', '[   0   4    4        6 ] common_prototypes1.h: ')


@isolate_nexus_core
def test_rmg_scalar_core_completion_requires_both_current_paths(tmp_path):

    sim = generate_rmg(
        job = job(machine='ws1', cores=1),
        output_wave_function_file = 'scf',
        write_mfr_potential = True,
        write_mfr_core_action = True,
        )
    sim.locdir = str(tmp_path)
    sim.job.finished = True
    for kind in ['orbitals', 'mfr_potential', 'mfr_core_action']:
        Path(sim.get_result(kind, None).h5file).write_bytes(b'old result')
    #end for
    base = ('potential convergence has been achieved. stopping ...\n'
            'TIMING INFORMATION FOR THREADS\n'
            'Wrote converged MFR potential to scf.mfr.h5\n')
    log = tmp_path/(sim.infile+'.00.log')
    log.write_text(base+'Wrote converged MFR core actions to other.mfr_core.h5\n')
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded
    log.write_text(base+'Wrote converged MFR core actions to scf.mfr_core.h5\n')
    sim.check_sim_status()
    assert sim.succeeded and not sim.failed
#end def test_rmg_scalar_core_completion_requires_both_current_paths


@isolate_nexus_core
def test_rmg_mfr_completion_uses_numeric_latest_log(tmp_path):
    sim = generate_rmg(
        job = job(machine='ws1', cores=1),
        output_wave_function_file = 'scf',
        write_mfr_core_action = True,
        )
    sim.locdir = str(tmp_path)
    sim.job.finished = True
    for kind in ['orbitals', 'mfr_core_action']:
        Path(sim.get_result(kind, None).h5file).write_bytes(b'old result')
    #end for
    complete = ('potential convergence has been achieved. stopping ...\n'
                'Wrote converged MFR core actions to scf.mfr_core.h5\n'
                'TIMING INFORMATION FOR THREADS\n')
    (tmp_path/(sim.infile+'.99.log')).write_text(complete)
    sim.check_sim_status()
    assert sim.succeeded and not sim.failed
    latest = tmp_path/(sim.infile+'.100.log')
    latest.write_text('native retry failed\n')
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded
    latest.write_text(complete)
    (tmp_path/(sim.infile+'.999extra.log')).write_text('unrelated file\n')
    sim.check_sim_status()
    assert sim.succeeded and not sim.failed
#end def test_rmg_mfr_completion_uses_numeric_latest_log
