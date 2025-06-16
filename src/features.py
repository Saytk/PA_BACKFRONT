"""
Feature engineering module for OHLC candle data.

This module provides functions to generate features from OHLC data,
including technical indicators, statistical features, and temporal features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import talib


def add_technical_indicators(df: pd.DataFrame, indicators: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.
    """
    result_df = df.copy()

    if indicators is None:
        indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr']

    if 'sma' in indicators:
        for period in [5, 10, 20, 50, 200]:
            result_df[f'sma_{period}'] = talib.SMA(result_df['close'], timeperiod=period)

    if 'ema' in indicators:
        for period in [5, 10, 20, 50, 200]:
            result_df[f'ema_{period}'] = talib.EMA(result_df['close'], timeperiod=period)

    if 'rsi' in indicators:
        result_df['rsi_14'] = talib.RSI(result_df['close'], timeperiod=14)

    if 'macd' in indicators:
        macd, macdsignal, macdhist = talib.MACD(
            result_df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        result_df['macd'] = macd
        result_df['macd_signal'] = macdsignal
        result_df['macd_hist'] = macdhist

    if 'bollinger' in indicators:
        upper, middle, lower = talib.BBANDS(
            result_df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        result_df['bollinger_upper'] = upper
        result_df['bollinger_middle'] = middle
        result_df['bollinger_lower'] = lower

    if 'atr' in indicators:
        result_df['atr_14'] = talib.ATR(result_df['high'], result_df['low'], result_df['close'], timeperiod=14)

    return result_df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features based on the timestamp.
    """
    result_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(result_df['date']):
        result_df['date'] = pd.to_datetime(result_df['date'])

    result_df['day_of_week'] = result_df['date'].dt.dayofweek
    result_df['day_of_month'] = result_df['date'].dt.day
    result_df['month'] = result_df['date'].dt.month

    result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
    result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)

    return result_df


def add_lagged_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Add lagged values of specified columns.
    """
    result_df = df.copy()
    for col in columns:
        for lag in lags:
            result_df[f'{col}_lag_{lag}'] = result_df[col].shift(lag)
    return result_df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return-based features (percent changes).
    """
    result_df = df.copy()
    result_df['return_1'] = result_df['close'].pct_change(periods=1)
    result_df['return_5'] = result_df['close'].pct_change(periods=5)
    result_df['log_return_1'] = np.log(result_df['close'] / result_df['close'].shift(1))
    result_df['volatility_5'] = result_df['return_1'].rolling(window=5).std()
    return result_df


def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add custom domain-specific features.
    """
    result_df = df.copy()
    result_df['body_size'] = np.abs(result_df['close'] - result_df['open'])
    result_df['upper_shadow'] = result_df['high'] - result_df[['close', 'open']].max(axis=1)
    result_df['lower_shadow'] = result_df[['close', 'open']].min(axis=1) - result_df['low']
    return result_df


def select_features(df: pd.DataFrame, feature_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Select specific features from the DataFrame.
    """
    if feature_list is None:
        return df
    return df[[f for f in feature_list if f in df.columns]]


if __name__ == "__main__":
    # Test zone (à compléter avec un petit CSV pour valider)
    pass
