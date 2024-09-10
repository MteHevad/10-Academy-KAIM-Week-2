import pytest
from src.data_preprocessing import load_and_clean_data, handle_outliers

def test_load_and_clean_data():
    df = load_and_clean_data('Week2_challenge_data_source.csv')
    assert df.isnull().sum().sum() == 0

def test_handle_outliers():
    df = load_and_clean_data('Week2_challenge_data_source.csv')
    df_clean = handle_outliers(df)
    assert df_clean.isnull().sum().sum() == 0
