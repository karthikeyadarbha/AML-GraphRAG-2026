import pandas as pd
import pytest

from scripts.data_pipeline.phase4_statistical_validation import validate_dataset_rigor


def make_base_ledger():
    return pd.DataFrame([
        {
            'Source_Account': 'A',
            'Target_Account': 'B',
            'Amount': 100.0,
            'Is_Synthetic_Fraud': False,
            'Risk_Level': 'LOW_RISK',
        },
        {
            'Source_Account': 'B',
            'Target_Account': 'C',
            'Amount': 200.0,
            'Is_Synthetic_Fraud': False,
            'Risk_Level': 'MEDIUM_RISK',
        },
        {
            'Source_Account': 'C',
            'Target_Account': 'D',
            'Amount': 300.0,
            'Is_Synthetic_Fraud': True,
            'Risk_Level': 'HIGH_RISK',
        },
        {
            'Source_Account': 'D',
            'Target_Account': 'E',
            'Amount': 400.0,
            'Is_Synthetic_Fraud': True,
            'Risk_Level': 'CRITICAL_RISK',
        },
    ])


def test_validate_dataset_rigor_passes_with_all_benchmark_levels():
    df = make_base_ledger()
    validate_dataset_rigor(df)


def test_validate_dataset_rigor_fails_when_risk_level_missing():
    df = make_base_ledger().drop(columns=['Risk_Level'])
    with pytest.raises(AssertionError, match="Risk_Level column missing"):
        validate_dataset_rigor(df)


def test_validate_dataset_rigor_fails_when_benchmark_level_missing():
    df = make_base_ledger().query("Risk_Level != 'CRITICAL_RISK'")
    with pytest.raises(AssertionError, match="Missing benchmark risk levels"):
        validate_dataset_rigor(df)
