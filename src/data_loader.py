# src/data_loader.py

import pandas as pd
from typing import Tuple

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load raw OHLC CSV data.
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    return df

def train_test_split_time_series(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and test sets while preserving temporal order.
    """
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]

# TODO: add more advanced preprocessing logic if needed
