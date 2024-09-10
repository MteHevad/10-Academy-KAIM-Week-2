import pytest
import pandas as pd
from src.clustering import perform_clustering

def test_perform_clustering():
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [10, 11, 12, 13]
    })
    clusters = perform_clustering(df, ['feature1', 'feature2'], n_clusters=2)
    assert len(clusters) == len(df)
