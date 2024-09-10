import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

def handle_outliers(df, threshold=3):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        df.loc[(df[col] > col_mean + threshold * col_std) | (df[col] < col_mean - threshold * col_std), col] = col_mean
    return df
