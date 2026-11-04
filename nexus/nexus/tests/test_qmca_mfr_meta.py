"""Focused qmca arithmetic tests for raw META field observables."""
import importlib.util
import sys
from importlib.machinery import SourceFileLoader
import numpy as np
import pytest

from . import TEST_DIR


QMCA_EXE = TEST_DIR.parent / 'bin/qmca'
_QMCA_LOADER = SourceFileLoader('qmca_meta_test_module', str(QMCA_EXE))
_QMCA_SPEC = importlib.util.spec_from_loader(_QMCA_LOADER.name, _QMCA_LOADER)
assert _QMCA_SPEC is not None and _QMCA_SPEC.loader is not None
qmca = importlib.util.module_from_spec(_QMCA_SPEC)
sys.modules[_QMCA_SPEC.name] = qmca
_QMCA_SPEC.loader.exec_module(qmca)


def write_scalar(path, columns, rows):
    with path.open('w') as stream:
        stream.write('# index ' + ' '.join(columns) + '\n')
        for index, row in enumerate(rows):
            values = [str(index)] + [f'{row[column]:.12g}' for column in columns]
            stream.write(' '.join(values) + '\n')


def analyze_file(path):
    analyzer = qmca.DatAnalyzer(str(path), index=0)
    analyzer.options = qmca.obj(equilibration=0)
    analyzer.autocorr = True  # Use DatAnalyzer.stat_value's simstats branch.
    analyzer.analyze()
    return analyzer


def meta_action_rows():
    rows = []
    for x in (-3, -2, -1, 0, 1, 2, 3, 4):
        rows.append({
            'LocalEnergy': -7.0,
            'TotalEnergyMF': -10.0,
            'KineticJastrowResidual': 1.0 + 2.0 * x,
            'ElecElec': 5.0 - x,
            'MFRMetaU': 2.0 + 3.0 * x,
            'MFRMetaDifferential': 4.0 - 2.0 * x,
            'MFRMetaDifferentialImag': 0.1 * (x % 2),
            'MFRCoreAction': -12.0 - 2.0 * x,
            'MFRCoreActionImag': 0.0,
            'IonIonMF': 0.5,
            'BlockWeight': 1.0,
        })
    return rows


def test_meta_energy_is_combined_per_block_and_keeps_covariance(tmp_path):
    columns = [
        'LocalEnergy', 'TotalEnergyMF', 'KineticJastrowResidual', 'ElecElec',
        'MFRMetaU', 'MFRMetaDifferential', 'MFRMetaDifferentialImag',
        'MFRCoreAction', 'MFRCoreActionImag', 'IonIonMF', 'BlockWeight',
    ]
    scalar = tmp_path / 'mfr_meta.s000.scalar.dat'
    write_scalar(scalar, columns, meta_action_rows())

    analyzer = analyze_file(scalar)
    data = analyzer.data
    stats = analyzer.stats

    # The four fluctuating META/Q terms cancel block by block. simstats sees
    # a constant derived series instead of independent means/errors.
    np.testing.assert_allclose(data.LocalEnergyMFRMeta, -10.0, atol=0.0)
    assert stats.LocalEnergyMFRMeta.mean == pytest.approx(-10.0)
    assert stats.LocalEnergyMFRMeta.var == pytest.approx(0.0, abs=1e-14)
    assert stats.LocalEnergyMFRMeta.error == pytest.approx(0.0, abs=1e-14)
    assert stats.MFRMetaU.var > 0.0

    expected_action = data.MFRCoreAction + data.KineticJastrowResidual + data.ElecElec + data.IonIonMF
    np.testing.assert_allclose(data.LocalEnergyMFR, expected_action)
    np.testing.assert_allclose(
        data.MetaMinusAction,
        data.LocalEnergyMFRMeta - data.LocalEnergyMFR,
    )
    assert stats.MetaMinusAction.mean == pytest.approx(-4.0)
    assert stats.MetaMinusAction.error > 0.0
    np.testing.assert_allclose(data.LocalEnergy, -7.0)

    assert qmca.QBase.quantity_abbreviations.emeta == 'LocalEnergyMFRMeta'
    assert qmca.QBase.quantity_abbreviations.dmeta == 'MetaMinusAction'
    assert qmca.QBase.quantity_abbreviations.m_u == 'MFRMetaU'
    assert qmca.QBase.quantity_abbreviations.m_d == 'MFRMetaDifferential'
    assert qmca.QBase.quantity_abbreviations.m_di == 'MFRMetaDifferentialImag'
    assert (
        qmca.DatAnalyzer.first.index('LocalEnergyMFR')
        < qmca.DatAnalyzer.first.index('LocalEnergyMFRMeta')
        < qmca.DatAnalyzer.first.index('MetaMinusAction')
    )


def test_meta_only_file_derives_energy_without_action_difference(tmp_path):
    columns = [
        'LocalEnergy', 'TotalEnergyMF', 'KineticJastrowResidual', 'ElecElec',
        'MFRMetaU', 'MFRMetaDifferential', 'BlockWeight',
    ]
    rows = [
        {
            'LocalEnergy': -7.0,
            'TotalEnergyMF': -10.0,
            'KineticJastrowResidual': 1.0 + 2.0 * x,
            'ElecElec': 5.0 - x,
            'MFRMetaU': 2.0 + 3.0 * x,
            'MFRMetaDifferential': 4.0 - 2.0 * x,
            'BlockWeight': 1.0,
        }
        for x in (-3, -2, -1, 0, 1, 2, 3, 4)
    ]
    scalar = tmp_path / 'meta_only.s000.scalar.dat'
    write_scalar(scalar, columns, rows)

    analyzer = analyze_file(scalar)
    assert analyzer.stats.LocalEnergyMFRMeta.mean == pytest.approx(-10.0)
    assert 'LocalEnergyMFR' not in analyzer.data
    assert 'MetaMinusAction' not in analyzer.data
    assert 'MetaMinusAction' not in analyzer.stats


def test_external_local_energy_mfr_is_not_mistaken_for_core_action(tmp_path):
    columns = [
        'LocalEnergy', 'LocalEnergyMFR', 'TotalEnergyMF',
        'KineticJastrowResidual', 'ElecElec', 'MFRMetaU',
        'MFRMetaDifferential', 'BlockWeight',
    ]
    rows = [
        {
            'LocalEnergy': -7.0,
            'LocalEnergyMFR': -50.0 - x,
            'TotalEnergyMF': -10.0,
            'KineticJastrowResidual': 1.0 + 2.0 * x,
            'ElecElec': 5.0 - x,
            'MFRMetaU': 2.0 + 3.0 * x,
            'MFRMetaDifferential': 4.0 - 2.0 * x,
            'BlockWeight': 1.0,
        }
        for x in (-3, -2, -1, 0, 1, 2, 3, 4)
    ]
    scalar = tmp_path / 'external_mfr_summary.s000.scalar.dat'
    write_scalar(scalar, columns, rows)

    analyzer = analyze_file(scalar)
    assert analyzer.stats.LocalEnergyMFRMeta.mean == pytest.approx(-10.0)
    assert analyzer.stats.LocalEnergyMFR.mean == pytest.approx(-50.5)
    assert 'MetaMinusAction' not in analyzer.data
    assert 'MetaMinusAction' not in analyzer.stats


def test_meta_core_action_requires_complete_action_companions(tmp_path):
    columns = [
        'LocalEnergy', 'TotalEnergyMF', 'KineticJastrowResidual',
        'ElecElec', 'MFRMetaU', 'MFRMetaDifferential',
        'MFRCoreAction', 'BlockWeight',
    ]
    rows = [
        {
            'LocalEnergy': -7.0,
            'TotalEnergyMF': -10.0,
            'KineticJastrowResidual': 1.0 + 2.0 * x,
            'ElecElec': 5.0 - x,
            'MFRMetaU': 2.0 + 3.0 * x,
            'MFRMetaDifferential': 4.0 - 2.0 * x,
            'MFRCoreAction': -12.0 - 2.0 * x,
            'BlockWeight': 1.0,
        }
        for x in (-3, -2, -1, 0, 1, 2, 3, 4)
    ]
    scalar = tmp_path / 'incomplete_core_meta.s000.scalar.dat'
    write_scalar(scalar, columns, rows)

    with pytest.raises(
        ValueError,
        match='incomplete MFR core/meta comparison input.*IonIonMF',
    ):
        analyze_file(scalar)


def test_incomplete_meta_columns_are_rejected_before_reporting(tmp_path):
    scalar = tmp_path / 'incomplete_meta.s000.scalar.dat'
    columns = [
        'LocalEnergy', 'TotalEnergyMF', 'KineticJastrowResidual', 'ElecElec',
        'MFRMetaU', 'BlockWeight',
    ]
    rows = [{
        'LocalEnergy': -7.0,
        'TotalEnergyMF': -10.0,
        'KineticJastrowResidual': 1.0,
        'ElecElec': 5.0,
        'MFRMetaU': 2.0,
        'BlockWeight': 1.0,
    } for _ in range(4)]
    write_scalar(scalar, columns, rows)

    with pytest.raises(ValueError, match='incomplete MFR meta input.*MFRMetaDifferential'):
        analyze_file(scalar)


def test_meta_requires_shared_scalar_formula_inputs(tmp_path):
    scalar = tmp_path / 'missing_meta_support.s000.scalar.dat'
    columns = [
        'LocalEnergy', 'TotalEnergyMF', 'KineticJastrowResidual',
        'MFRMetaU', 'MFRMetaDifferential', 'BlockWeight',
    ]
    rows = [{
        'LocalEnergy': -7.0,
        'TotalEnergyMF': -10.0,
        'KineticJastrowResidual': 1.0,
        'MFRMetaU': 2.0,
        'MFRMetaDifferential': 4.0,
        'BlockWeight': 1.0,
    } for _ in range(4)]
    write_scalar(scalar, columns, rows)

    with pytest.raises(ValueError, match='incomplete MFR meta input.*ElecElec'):
        analyze_file(scalar)


def test_scalar_and_meta_columns_are_rejected_together(tmp_path):
    scalar = tmp_path / 'scalar_and_meta.s000.scalar.dat'
    columns = [
        'LocalEnergy', 'TotalEnergyMF', 'KineticJastrowResidual', 'ElecElec',
        'MFRPotential', 'MFRMetaU', 'MFRMetaDifferential', 'BlockWeight',
    ]
    rows = [{
        'LocalEnergy': -7.0,
        'TotalEnergyMF': -10.0,
        'KineticJastrowResidual': 1.0,
        'ElecElec': 5.0,
        'MFRPotential': 2.0,
        'MFRMetaU': 2.0,
        'MFRMetaDifferential': 4.0,
        'BlockWeight': 1.0,
    } for _ in range(4)]
    write_scalar(scalar, columns, rows)

    with pytest.raises(ValueError, match='scalar MFRPotential and MFR meta fields'):
        analyze_file(scalar)


def test_legacy_scalar_and_core_action_arithmetic_is_unchanged(tmp_path):
    scalar = tmp_path / 'legacy_scalar.s000.scalar.dat'
    scalar_columns = [
        'LocalEnergy', 'TotalEnergyMF', 'KineticJastrowResidual',
        'ElecElec', 'MFRPotential', 'BlockWeight',
    ]
    scalar_rows = [
        {'LocalEnergy': -7.0, 'TotalEnergyMF': -10.0,
         'KineticJastrowResidual': float(i + 1), 'ElecElec': float(i + 5),
         'MFRPotential': float(2 * (i + 1)), 'BlockWeight': 1.0}
        for i in range(4)
    ]
    write_scalar(scalar, scalar_columns, scalar_rows)
    scalar_analyzer = analyze_file(scalar)
    assert scalar_analyzer.stats.LocalEnergyMFR.mean == pytest.approx(-6.0)
    np.testing.assert_allclose(scalar_analyzer.data.LocalEnergy, -7.0)

    action = tmp_path / 'legacy_action.s000.scalar.dat'
    action_columns = [
        'LocalEnergy', 'ElecElec', 'KineticJastrowResidual',
        'MFRCoreAction', 'MFRCoreActionImag', 'IonIonMF', 'BlockWeight',
    ]
    action_rows = [
        {'LocalEnergy': -7.0, 'ElecElec': float(i + 5),
         'KineticJastrowResidual': float(i + 1),
         'MFRCoreAction': float(-12 + i), 'MFRCoreActionImag': 0.0,
         'IonIonMF': 0.5, 'BlockWeight': 1.0}
        for i in range(4)
    ]
    write_scalar(action, action_columns, action_rows)
    action_analyzer = analyze_file(action)
    assert action_analyzer.stats.LocalEnergyMFR.mean == pytest.approx(-1.0)
    np.testing.assert_allclose(action_analyzer.data.LocalEnergy, -7.0)
