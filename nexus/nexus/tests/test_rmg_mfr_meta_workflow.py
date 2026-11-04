"""RMG meta-field input/result and QMCPACK dependency plumbing tests."""
import os
from pathlib import Path

import pytest

from . import create_pseudo_files, isolate_nexus_core


def helium_system():
    from ..physical_system import generate_physical_system

    return generate_physical_system(
        axes=[[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        elem=["He"],
        pos=[[0, 0, 0]],
        units="B",
        kgrid=(1, 1, 1),
        kshift=(0, 0, 0),
    )


@isolate_nexus_core
def test_rmg_meta_input_flags_results_and_q_source_guard(tmp_path):
    from ..machines import job
    from ..rmg import generate_rmg
    from ..rmg_input import RmgInput, input_spec

    assert input_spec.keywords.write_mfr_meta_fields.default is False
    assert input_spec.keywords.mfr_meta_q_source.default == "functional_derivative"
    assert input_spec.keywords.mfr_meta_q_source.allowed == {
        "functional_derivative",
        "effective_smooth",
    }

    sim = generate_rmg(
        job=job(machine="ws1", cores=1),
        output_wave_function_file="Waves/meta",
        write_mfr_meta_fields=True,
        mfr_meta_q_source="effective_smooth",
    )
    sim.locdir = str(tmp_path)

    assert sim.check_result("mfr_meta_fields", None)
    assert sim.check_result("orbitals", None)
    assert not sim.check_result("mfr_core_action", None)
    assert not sim.check_result("mfr_potential", None)
    assert sim.get_result("mfr_meta_fields", None).h5file == str(
        tmp_path / "Waves/meta.mfr_meta.h5"
    )
    assert sim.get_result("orbitals", None).h5file == str(tmp_path / "Waves/meta.h5")
    assert sim.get_output_files() == [
        "Waves/meta.mfr_meta.h5",
        "Waves/meta.h5",
    ]

    restored = RmgInput()
    text = sim.input.write_text()
    restored.read_text(text)
    assert restored.write_mfr_meta_fields is True
    assert restored.mfr_meta_q_source == "effective_smooth"

    invalid = generate_rmg(
        identifier="meta_bad_source",
        path="meta_bad_source",
        job=job(machine="ws1", cores=1),
        write_mfr_meta_fields=True,
        mfr_meta_q_source="unsupported_source",
    )
    assert not invalid.input.is_valid()
    with pytest.raises(ValueError, match="mfr_meta_q_source"):
        invalid.input.write_text()
#end def test_rmg_meta_input_flags_results_and_q_source_guard


@isolate_nexus_core
def test_rmg_meta_completion_requires_current_marker_and_nonempty_result(tmp_path):
    from ..machines import job
    from ..rmg import generate_rmg

    sim = generate_rmg(
        job=job(machine="ws1", cores=1),
        output_wave_function_file="Waves/meta",
        write_mfr_meta_fields=True,
    )
    sim.locdir = str(tmp_path)
    sim.job.finished = True

    log00 = tmp_path / (sim.infile + ".00.log")
    log01 = tmp_path / (sim.infile + ".01.log")
    meta_file = Path(sim.get_result("mfr_meta_fields", None).h5file)
    orbital_file = Path(sim.get_result("orbitals", None).h5file)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    base = (
        "potential convergence has been achieved. stopping ...\n"
        "TIMING INFORMATION FOR THREADS\n"
    )

    log00.write_text(base)
    meta_file.write_bytes(b"old meta")
    orbital_file.write_bytes(b"old orbitals")
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded  # Old files alone cannot complete a current run.

    log00.write_text(base + "Wrote converged MFR meta fields to stale.mfr_meta.h5\n")
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded  # Marker must name this result path.

    log00.write_text(
        base + "Wrote converged MFR meta fields to Waves/meta.mfr_meta.h5\n"
    )
    meta_file.write_bytes(b"")
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded  # Empty output is not a result.

    meta_file.write_bytes(b"current meta result")
    sim.check_sim_status()
    assert sim.succeeded and not sim.failed

    prefix = "[   0   4    4        6 ] common_prototypes1.h: "
    log00.write_text(base + prefix + "Wrote converged MFR meta fields to Waves/meta.mfr_meta.h5\n")
    sim.check_sim_status()
    assert sim.succeeded and not sim.failed  # Actual native meta-only marker form.
    log00.write_text(base + prefix + "Wrote converged MFR meta fields to other/meta.mfr_meta.h5\n")
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded
    log00.write_text(base + prefix + "Wrote converged MFR meta fields to Waves/meta.mfr_meta.h5\n")

    log01.write_text(base)
    sim.check_sim_status()
    assert sim.failed and not sim.succeeded  # A retry needs its own current marker.
#end def test_rmg_meta_completion_requires_current_marker_and_nonempty_result


@isolate_nexus_core
def test_rmg_meta_qmcpack_dependency_requires_provider_and_nonempty_file(tmp_path):
    from ..machines import job
    from ..qmcpack import generate_qmcpack
    from ..qmcpack_input import vmc
    from ..rmg import generate_rmg

    create_pseudo_files(tmp_path, ["He.BFD.xml"])
    rmg = generate_rmg(
        job=job(machine="ws1", cores=1),
        output_wave_function_file="scf",
        write_mfr_meta_fields=True,
    )
    rmg.locdir = str(tmp_path)
    meta_result = rmg.get_result("mfr_meta_fields", None)
    Path(meta_result.h5file).write_bytes(b"complete meta fixture")

    qmc = generate_qmcpack(
        identifier="meta_qmc",
        path=str(tmp_path / "qmc"),
        job=job(machine="ws1", cores=1),
        system=helium_system(),
        pseudos=["He.BFD.xml"],
        jastrows=[],
        mfr_meta_fields="placeholder.mfr_meta.h5",
        mfr_qmc=True,
        check_paths=False,
        calculations=[vmc(blocks=1, steps=1)],
    )
    qmc.incorporate_result("mfr_meta_fields", meta_result, rmg)
    assert qmc.input.get("MFRMetaFields").href == os.path.relpath(
        meta_result.h5file, qmc.locdir
    )
    assert 'type="mfr_meta_fields"' in qmc.input.write_text()

    missing_provider = generate_qmcpack(
        identifier="without_meta_provider",
        path=str(tmp_path / "without_meta_provider"),
        job=job(machine="ws1", cores=1),
        system=helium_system(),
        pseudos=["He.BFD.xml"],
        jastrows=[],
        mfr_qmc=False,
        check_paths=False,
        calculations=[vmc(blocks=1, steps=1)],
    )
    with pytest.raises(ValueError, match="requires mfr_meta_fields"):
        missing_provider.incorporate_result("mfr_meta_fields", meta_result, rmg)

    disabled = generate_rmg(
        identifier="meta_disabled",
        path="meta_disabled",
        job=job(machine="ws1", cores=1),
        output_wave_function_file="disabled",
    )
    with pytest.raises(ValueError, match="write_mfr_meta_fields=True"):
        qmc.incorporate_result("mfr_meta_fields", meta_result, disabled)

    Path(meta_result.h5file).write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        qmc.incorporate_result("mfr_meta_fields", meta_result, rmg)
#end def test_rmg_meta_qmcpack_dependency_requires_provider_and_nonempty_file


@isolate_nexus_core
def test_rmg_core_plus_meta_comparison_requires_explicit_provider_mode(tmp_path):
    from ..machines import job
    from ..qmcpack import generate_qmcpack
    from ..qmcpack_input import vmc

    create_pseudo_files(tmp_path, ["He.BFD.xml"])
    common = dict(
        job=job(machine="ws1", cores=1),
        system=helium_system(),
        pseudos=["He.BFD.xml"],
        jastrows=[],
        mfr_core_action="core.mfr_core.h5",
        mfr_meta_fields="meta.mfr_meta.h5",
        check_paths=False,
        calculations=[vmc(blocks=1, steps=1)],
    )
    with pytest.raises(ValueError, match="explicit mfr_qmc=False"):
        generate_qmcpack(identifier="ambiguous", path=str(tmp_path / "ambiguous"), **common)

    comparison = generate_qmcpack(
        identifier="comparison",
        path=str(tmp_path / "comparison"),
        mfr_qmc=False,
        **common,
    )
    assert comparison.input.get("MFRCoreAction") is not None
    assert comparison.input.get("MFRMetaFields") is not None
    assert comparison.input.simulation.calculations[0].mfr_qmc is False
#end def test_rmg_core_plus_meta_comparison_requires_explicit_provider_mode
