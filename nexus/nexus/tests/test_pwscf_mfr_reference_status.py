from . import isolate_nexus_core


@isolate_nexus_core
def test_control_write_mfr_reference_status_is_optional_boolean():
    from ..pwscf_input import PwscfInput
    from ..pwscf_input_defs import ControlDefinitions

    definition = ControlDefinitions.write_mfr_reference_status.value
    assert definition.input_name == 'write_mfr_reference_status'
    assert definition.datatype is bool
    assert definition.required is False

    for value, rendered in ((True, '.true.'), (False, '.false.')):
        pw = PwscfInput('control')
        pw.control.write_mfr_reference_status = value
        text = pw.control.write(pw)
        assert f'write_mfr_reference_status = {rendered}' in text

        restored = PwscfInput('control')
        restored.read_text(text)
        assert restored.control.write_mfr_reference_status is value
