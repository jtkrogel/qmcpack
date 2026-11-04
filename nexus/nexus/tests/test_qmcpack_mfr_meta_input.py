"""Focused Nexus generation tests for the QMCPACK mfr_meta_fields extpot."""
import os

import pytest

from . import isolate_nexus_core


def make_system():
    from ..physical_system import generate_physical_system

    return generate_physical_system(
        units='B',
        axes=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        elem_pos='He 0.0 0.0 0.0',
    )


def generate(system, **kwargs):
    from ..qmcpack_input import generate_qmcpack_input

    options = dict(
        input_type='basic',
        system=system,
        orbitals_h5='MISSING.h5',
        corrections=[],
        jastrows=[],
        calculations=[],
    )
    options.update(kwargs)
    return generate_qmcpack_input(**options)


@isolate_nexus_core
def test_meta_extpot_xml_collection_and_relative_path_roundtrip(tmp_path):
    from ..qmcpack_input import QmcpackInput

    system = make_system()
    meta_file = tmp_path / 'fields.mfr_meta.h5'
    meta_file.touch()
    run_path = tmp_path / 'qmc'
    run_path.mkdir()

    qi = generate(
        system,
        mfr_meta_fields=str(meta_file),
        mfr_qmc=False,
        run_path=str(run_path),
    )
    text = qi.write()
    expected_href = os.path.relpath(meta_file, run_path)

    assert '<extpot' in text
    assert 'type="mfr_meta_fields"' in text
    assert 'name="MFRMetaFields"' in text
    assert f'href="{expected_href}"' in text
    assert text.count('name="MFRMetaFields"') == 1

    input_path = tmp_path / 'generated.xml'
    input_path.write_text(text)
    parsed = QmcpackInput(input_path)
    parsed.pluralize()
    assert parsed.get('MFRMetaFields').href == expected_href


@isolate_nexus_core
def test_meta_provider_uses_existing_standalone_default_and_explicit_no_mode(tmp_path):
    system = make_system()
    meta_file = tmp_path / 'fields.mfr_meta.h5'
    meta_file.touch()

    standalone = generate(system, mfr_meta_fields=str(meta_file))
    standalone_text = standalone.write()
    assert 'type="mfr_meta_fields"' in standalone_text
    assert 'name="IonIon"' not in standalone_text
    assert 'name="ElecIon"' not in standalone_text

    explicit_standalone = generate(
        system,
        mfr_meta_fields=str(meta_file),
        mfr_qmc=True,
    )
    explicit_text = explicit_standalone.write()
    assert 'type="mfr_meta_fields"' in explicit_text
    assert 'name="IonIon"' not in explicit_text
    assert 'name="ElecIon"' not in explicit_text

    comparison = generate(
        system,
        mfr_meta_fields=str(meta_file),
        mfr_qmc=False,
    )
    comparison_text = comparison.write()
    assert 'type="mfr_meta_fields"' in comparison_text
    assert 'name="IonIon"' in comparison_text
    assert 'name="ElecIon"' in comparison_text


@isolate_nexus_core
def test_core_and_meta_require_explicit_provider_mode_and_scalar_meta_is_rejected(tmp_path):
    system = make_system()
    core_file = tmp_path / 'core.mfr_core.h5'
    meta_file = tmp_path / 'fields.mfr_meta.h5'
    scalar_file = tmp_path / 'scalar.mfr.h5'
    for path in (core_file, meta_file, scalar_file):
        path.touch()

    combined = generate(
        system,
        mfr_core_action=str(core_file),
        mfr_meta_fields=str(meta_file),
        mfr_qmc=False,
    )
    combined_text = combined.write()
    assert 'type="mfr_core_action"' in combined_text
    assert 'name="MFRCoreAction"' in combined_text
    assert 'type="mfr_meta_fields"' in combined_text
    assert 'name="MFRMetaFields"' in combined_text
    assert 'name="IonIon"' in combined_text
    assert 'name="ElecIon"' in combined_text

    for mode in (None, True):
        with pytest.raises(ValueError, match='require explicit mfr_qmc=False'):
            generate(
                system,
                mfr_core_action=str(core_file),
                mfr_meta_fields=str(meta_file),
                mfr_qmc=mode,
            )

    with pytest.raises(ValueError, match='mfr_potential and mfr_meta_fields'):
        generate(
            system,
            mfr_potential=str(scalar_file),
            mfr_meta_fields=str(meta_file),
            mfr_qmc=False,
        )


@isolate_nexus_core
def test_existing_core_and_scalar_standalone_defaults_remain_unchanged(tmp_path):
    system = make_system()
    core_file = tmp_path / 'core.mfr_core.h5'
    scalar_file = tmp_path / 'scalar.mfr.h5'
    core_file.touch()
    scalar_file.touch()

    for provider, path, xml_type in (
        ('mfr_core_action', core_file, 'mfr_core_action'),
        ('mfr_potential', scalar_file, 'mfr'),
    ):
        qi = generate(system, **{provider: str(path)})
        text = qi.write()
        assert f'type="{xml_type}"' in text
        assert 'name="IonIon"' not in text
        assert 'name="ElecIon"' not in text


@isolate_nexus_core
def test_meta_path_type_and_existence_are_checked(tmp_path):
    system = make_system()

    with pytest.raises(TypeError, match='mfr_meta_fields must be a file path'):
        generate(system, mfr_meta_fields=tmp_path / 'fields.mfr_meta.h5')

    with pytest.raises(FileNotFoundError, match='mfr_meta_fields'):
        generate(system, mfr_meta_fields=str(tmp_path / 'missing.mfr_meta.h5'))

    unchecked = generate(
        system,
        mfr_meta_fields=str(tmp_path / 'future.mfr_meta.h5'),
        check_paths=False,
    )
    assert 'type="mfr_meta_fields"' in unchecked.write()

    baseline = generate(system)
    baseline_text = baseline.write()
    assert '<extpot' not in baseline_text
    assert 'name="IonIon"' in baseline_text
    assert 'name="ElecIon"' in baseline_text
