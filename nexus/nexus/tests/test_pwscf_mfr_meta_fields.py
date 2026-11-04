import pytest

from . import isolate_nexus_core


@isolate_nexus_core
def test_write_mfr_meta_fields_control_definition_and_roundtrip():
    from ..pwscf_input import PwscfInput, check_section_classes
    from ..pwscf_input_defs import ControlDefinitions

    scalar_definition = ControlDefinitions.write_mfr_potential.value
    assert scalar_definition.input_name == 'write_mfr_potential'
    assert scalar_definition.datatype is bool
    assert scalar_definition.required is False

    definition = ControlDefinitions.write_mfr_meta_fields.value
    assert definition.input_name == 'write_mfr_meta_fields'
    assert definition.datatype is bool
    assert definition.required is False
    assert ControlDefinitions.write_mfr_reference_status.value.datatype is bool

    # Ensure the new CONTROL key is registered with the generated definitions.
    check_section_classes(exit=False)

    for value in (True, False):
        pw = PwscfInput('control')
        pw.control.write_mfr_potential = value
        pw.control.write_mfr_meta_fields = not value
        pw.control.write_mfr_reference_status = value
        text = pw.control.write(pw)
        rendered = '.true.' if value else '.false.'
        rendered_meta = '.false.' if value else '.true.'
        assert f'write_mfr_potential = {rendered}' in text
        assert f'write_mfr_meta_fields = {rendered_meta}' in text
        assert f'write_mfr_reference_status = {rendered}' in text

        restored = PwscfInput()
        restored.read_text(text)
        assert restored.control.write_mfr_potential is value
        assert restored.control.write_mfr_meta_fields is (not value)
        assert restored.control.write_mfr_reference_status is value


@isolate_nexus_core
def test_write_mfr_meta_fields_generated_control_and_key_validation():
    from ..developer import obj
    from ..pwscf_input import PwscfInput, generate_pwscf_input

    generated = generate_pwscf_input(
        selector='generic',
        calculation='scf',
        prefix='mfr_meta_fixture',
        outdir='./out',
        pseudo_dir='./pseudos',
        write_mfr_potential=True,
        write_mfr_meta_fields=True,
        write_mfr_reference_status=True,
        ibrav=1,
        celldm={1: 10.0},
        nat=1,
        ntyp=1,
        ecutwfc=30.0,
        elem=['He'],
        mass=obj(He=4.0),
        pos=[[0.0, 0.0, 0.0]],
        positions_option='crystal',
        pseudos=['He.upf'],
        kgrid=(1, 1, 1),
        kshift=(0, 0, 0),
    )
    generated_text = generated.write_text()
    assert 'write_mfr_potential = .true.' in generated_text
    assert 'write_mfr_meta_fields = .true.' in generated_text
    assert 'write_mfr_reference_status = .true.' in generated_text

    restored = PwscfInput()
    restored.read_text(generated_text)
    assert restored.control.write_mfr_potential is True
    assert restored.control.write_mfr_meta_fields is True
    assert restored.control.write_mfr_reference_status is True

    default_input = generate_pwscf_input(
        selector='generic',
        calculation='scf',
        prefix='mfr_meta_default',
        outdir='./out',
        pseudo_dir='./pseudos',
        ibrav=1,
        celldm={1: 10.0},
        nat=1,
        ntyp=1,
        ecutwfc=30.0,
        elem=['He'],
        mass=obj(He=4.0),
        pos=[[0.0, 0.0, 0.0]],
        positions_option='crystal',
        pseudos=['He.upf'],
        kgrid=(1, 1, 1),
        kshift=(0, 0, 0),
    )
    default_text = default_input.write_text()
    assert 'write_mfr_potential' not in default_text
    assert 'write_mfr_meta_fields' not in default_text

    malformed = PwscfInput()
    with pytest.raises(KeyError, match='write_mfr_meta_field'):
        malformed.read_text(
            '&CONTROL\n'
            '  write_mfr_meta_field = .true.\n'
            '/\n'
        )
