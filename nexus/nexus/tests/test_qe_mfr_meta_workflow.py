"""Focused Nexus plumbing tests for QE's paired MFR meta-fields export."""
import os
from pathlib import Path

import pytest

from . import create_pseudo_files, isolate_nexus_core


@isolate_nexus_core
def test_pw2mfr_core_meta_input_and_conditional_results(tmp_path):
    from ..machines import job
    from ..developer import obj
    from ..qmcpack_converters import Pw2mfrCoreInput, generate_pw2mfr_core
    from ..qmcpack_converters import generate_pw2mfr_core_input

    inp = generate_pw2mfr_core_input(
        prefix='carbon',
        outdir='pwscf_output',
        last_band=4,
        write_diagnostics=True,
        write_meta_fields=True,
        meta_q_source='effective_smooth',
        )
    written = tmp_path/'pw2mfr_core.in'
    inp.write(written)
    restored = Pw2mfrCoreInput(written)
    assert restored.inputpp.prefix == 'carbon'
    assert restored.inputpp.outdir == 'pwscf_output'
    assert restored.inputpp.last_band == 4
    assert restored.inputpp.write_diagnostics
    assert restored.inputpp.write_meta_fields
    assert restored.inputpp.meta_q_source == 'effective_smooth'

    exporter = generate_pw2mfr_core(
        input=inp,
        identifier='meta_export',
        path=str(tmp_path/'export'),
        job=job(machine='ws1',cores=1,app='pw2mfr_core.x'),
        )
    core = exporter.get_result('mfr_core_action',None)
    meta = exporter.get_result('mfr_meta_fields',None)
    assert core.h5file == str(tmp_path/'export'/'pwscf_output'/'carbon.mfr_core.h5')
    assert meta.h5file == str(tmp_path/'export'/'pwscf_output'/'carbon.mfr_meta.h5')
    assert exporter.check_result('mfr_core_action',None)
    assert exporter.check_result('mfr_meta_fields',None)
    assert not exporter.check_result('unknown',None)
    exporter.produces = set()
    exporter.fill_produces()
    assert 'mfr_meta_fields' in exporter.produces
    exporter.products = obj()
    exporter.fill_products()
    assert exporter.products.mfr_meta_fields == meta.h5file
    assert exporter.get_output_files() == [
        os.path.relpath(core.h5file,exporter.locdir),
        os.path.relpath(meta.h5file,exporter.locdir),
        ]

    core_only = generate_pw2mfr_core(
        identifier='core_only',
        path=str(tmp_path/'core_only'),
        job=job(machine='ws1',cores=1,app='pw2mfr_core.x'),
        )
    assert not core_only.check_result('mfr_meta_fields',None)
    core_only.produces = set()
    core_only.fill_produces()
    assert 'mfr_meta_fields' not in core_only.produces
    assert core_only.get_output_files() == [
        os.path.relpath(
            core_only.get_result('mfr_core_action',None).h5file,
            core_only.locdir,
            )
        ]
    with pytest.raises(NotImplementedError,match='only when write_meta_fields is enabled'):
        core_only.get_result('mfr_meta_fields',None)

    default_input = generate_pw2mfr_core_input()
    assert default_input.inputpp.write_meta_fields is False
    assert default_input.inputpp.meta_q_source == 'functional_derivative'
    with pytest.raises(ValueError,match='meta_q_source must be'):
        generate_pw2mfr_core_input(meta_q_source='unknown_source')
#end def test_pw2mfr_core_meta_input_and_conditional_results


@isolate_nexus_core
def test_pw2mfr_core_requested_meta_status_requires_both_nonempty_current_exports(tmp_path):
    from ..machines import job
    from ..qmcpack_converters import generate_pw2mfr_core

    exporter = generate_pw2mfr_core(
        identifier='mfr_meta',
        path=str(tmp_path),
        job=job(machine='ws1',cores=1,app='pw2mfr_core.x'),
        write_meta_fields=True,
        )
    exporter.locdir = str(tmp_path)
    core_path = Path(exporter.get_result('mfr_core_action',None).h5file)
    meta_path = Path(exporter.get_result('mfr_meta_fields',None).h5file)
    core_path.parent.mkdir(parents=True,exist_ok=True)
    core_path.write_bytes(b'current core result')
    meta_path.write_bytes(b'current meta result')
    output_path = tmp_path/exporter.outfile
    exporter.job.finished = True

    good_output = (
        f'Bare MFR core actions written to {core_path.relative_to(tmp_path)}\n'
        f'Paired MFR meta fields written to {meta_path.relative_to(tmp_path)}\n'
        'JOB DONE.\n'
        )
    output_path.write_text(good_output)
    exporter.check_sim_status()
    assert exporter.finished and exporter.succeeded and not exporter.failed

    meta_path.unlink()
    exporter.check_sim_status()
    assert exporter.failed and not exporter.finished
    meta_path.write_bytes(b'current meta result')

    output_path.write_text(
        f'Bare MFR core actions written to {core_path.relative_to(tmp_path)}\nJOB DONE.\n'
        )
    exporter.check_sim_status()
    assert exporter.failed and not exporter.finished

    output_path.write_text(
        f'Bare MFR core actions written to {core_path.relative_to(tmp_path)}\n'
        'Paired MFR meta fields written to stale.mfr_meta.h5\n'
        'JOB DONE.\n'
        )
    exporter.check_sim_status()
    assert exporter.failed and not exporter.finished

    # A same-named result reported in a different directory is not the
    # requested current export, even when old files still exist here.
    output_path.write_text(good_output.replace(
        str(meta_path.relative_to(tmp_path)), 'other_directory/'+meta_path.name
        ))
    exporter.check_sim_status()
    assert exporter.failed and not exporter.finished

    # A prior completed HDF5 pair cannot make a failed retry successful.
    output_path.write_text('Converter failed before writing either result\n')
    exporter.check_sim_status()
    assert exporter.failed and not exporter.finished and not exporter.succeeded

    output_path.write_text(good_output)
    meta_path.write_bytes(b'')
    exporter.check_sim_status()
    assert exporter.failed and not exporter.finished

    meta_path.write_bytes(b'current meta result')
    exporter.check_sim_status()
    assert exporter.finished and exporter.succeeded and not exporter.failed
#end def test_pw2mfr_core_requested_meta_status_requires_both_nonempty_current_exports


@isolate_nexus_core
def test_qmcpack_qe_meta_dependency_paths_provider_and_missing_file(tmp_path):
    from ..machines import job
    from ..nexus_base import nexus_config
    from ..qmcpack import generate_qmcpack
    from ..qmcpack_converters import generate_pw2mfr_core
    from ..qmcpack_input import vmc
    from .test_pwscf_simulation import get_pwscf_sim

    nexus_config.runs = ''
    nexus_config.local_directory = str(tmp_path)
    nexus_config.remote_directory = str(tmp_path)
    create_pseudo_files(tmp_path,['C.BFD.upf','C.BFD.xml'])
    scf = get_pwscf_sim('scf')
    exporter = generate_pw2mfr_core(
        identifier='meta_export',
        path='meta_export',
        job=job(machine='ws1',cores=1,app='pw2mfr_core.x'),
        write_meta_fields=True,
        )
    core_result = exporter.get_result('mfr_core_action',None)
    meta_result = exporter.get_result('mfr_meta_fields',None)
    Path(core_result.h5file).parent.mkdir(parents=True,exist_ok=True)
    Path(core_result.h5file).write_bytes(b'core')
    Path(meta_result.h5file).write_bytes(b'meta')

    qmc = generate_qmcpack(
        identifier='qmc',
        path='qmc',
        job=job(machine='ws1',cores=1,app='qmcpack'),
        system=scf.system,
        pseudos=['C.BFD.xml'],
        jastrows=[],
        mfr_core_action='placeholder.mfr_core.h5',
        mfr_meta_fields='placeholder.mfr_meta.h5',
        mfr_qmc=False,
        check_paths=False,
        calculations=[vmc(blocks=1,steps=1)],
        )
    qmc.incorporate_result('mfr_core_action',core_result,exporter)
    qmc.incorporate_result('mfr_meta_fields',meta_result,exporter)
    assert qmc.input.get('MFRCoreAction').href == os.path.relpath(
        core_result.h5file,qmc.locdir)
    assert qmc.input.get('MFRMetaFields').href == os.path.relpath(
        meta_result.h5file,qmc.locdir)
    qmc_text = qmc.input.write_text()
    assert 'type="mfr_core_action"' in qmc_text
    assert 'type="mfr_meta_fields"' in qmc_text

    no_meta_provider = generate_qmcpack(
        identifier='qmc_no_meta_provider',
        path='qmc_no_meta_provider',
        job=job(machine='ws1',cores=1,app='qmcpack'),
        system=scf.system,
        pseudos=['C.BFD.xml'],
        jastrows=[],
        mfr_core_action='placeholder.mfr_core.h5',
        mfr_qmc=False,
        check_paths=False,
        calculations=[vmc(blocks=1,steps=1)],
        )
    with pytest.raises(ValueError,match='requires mfr_meta_fields'):
        no_meta_provider.incorporate_result('mfr_meta_fields',meta_result,exporter)

    Path(meta_result.h5file).unlink()
    with pytest.raises(FileNotFoundError):
        qmc.incorporate_result('mfr_meta_fields',meta_result,exporter)
#end def test_qmcpack_qe_meta_dependency_paths_provider_and_missing_file


@isolate_nexus_core
def test_pw2mfr_core_orbital_dependency_rejects_prefix_or_outdir_mismatch(tmp_path):
    from ..machines import job
    from ..qmcpack_converters import generate_pw2mfr_core
    from .test_pwscf_simulation import get_pwscf_sim

    create_pseudo_files(tmp_path,['C.BFD.upf'])
    scf = get_pwscf_sim('scf')
    exporter = generate_pw2mfr_core(
        prefix='different_prefix',
        job=job(machine='ws1',cores=1,app='pw2mfr_core.x'),
        )
    with pytest.raises(RuntimeError,match='same prefix and outdir'):
        exporter.incorporate_result('orbitals',scf.get_result('orbitals',exporter),scf)
#end def test_pw2mfr_core_orbital_dependency_rejects_prefix_or_outdir_mismatch


@isolate_nexus_core
def test_meta_dependency_requires_native_scf_snapshot_flag(tmp_path):
    from ..machines import job
    from ..qmcpack_converters import generate_pw2mfr_core
    from .test_pwscf_simulation import get_pwscf_sim

    create_pseudo_files(tmp_path,['C.BFD.upf'])
    scf = get_pwscf_sim('scf')
    control = scf.input.control
    exporter = generate_pw2mfr_core(
        prefix=control.prefix if 'prefix' in control else 'pwscf',
        outdir=control.outdir if 'outdir' in control else './',
        write_meta_fields=True,
        job=job(machine='ws1',cores=1,app='pw2mfr_core.x'),
        )
    exporter.locdir = scf.locdir
    result = scf.get_result('orbitals',exporter)
    with pytest.raises(RuntimeError,match='write_mfr_meta_fields=True'):
        exporter.incorporate_result('orbitals',result,scf)
    control.write_mfr_meta_fields = True
    exporter.incorporate_result('orbitals',result,scf)
#end def test_meta_dependency_requires_native_scf_snapshot_flag
