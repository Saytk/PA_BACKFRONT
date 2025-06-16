"""
Central configuration module for the OHLC prediction project.

This module contains all configuration parameters, paths, and settings
used throughout the project.
"""

import os
from typing import Dict, List, Any, Optional

# Project root directory (adjust if needed)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_PATH = os.path.join(ROOT_DIR, 'data')
MODELS_PATH = os.path.join(ROOT_DIR, 'models')
NOTEBOOKS_PATH = os.path.join(ROOT_DIR, 'notebooks')

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(NOTEBOOKS_PATH, exist_ok=True)

# Data configuration
DEFAULT_TRAIN_FILE = os.path.join(DATA_PATH, 'gold_ohlc_train.csv')
DEFAULT_TEST_FILE = os.path.join(DATA_PATH, 'gold_ohlc_test.csv')

# Column names
TIMESTAMP_COL = 'timestamp'
OHLC_COLUMNS = ['open', 'high', 'low', 'close']
VOLUME_COL = 'volume'  # If available

# Feature engineering configuration
DEFAULT_TECHNICAL_INDICATORS = [
    'sma',  # Simple Moving Average
    'ema',  # Exponential Moving Average
    'rsi',  # Relative Strength Index
    'macd',  # Moving Average Convergence Divergence
    'bollinger',  # Bollinger Bands
    'atr',  # Average True Range
]

# Default periods for technical indicators
INDICATOR_PERIODS = {
    'sma': [5, 10, 20, 50, 200],
    'ema': [5, 10, 20, 50, 200],
    'rsi': [14],
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
    'bollinger': {'period': 20, 'std_dev': 2},
    'atr': [14],
}

# Default lag periods for creating lagged features
DEFAULT_LAG_PERIODS = [1, 2, 3, 5, 10]

# Model configuration
DEFAULT_MODEL_TYPE = 'xgboost'

# Default model parameters for each model type
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42,
    },
    'lightgbm': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'regression',
        'random_state': 42,
    },
    'lstm': {
        'units': 50,
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'patience': 10,
    }
}

# Training configuration
TRAIN_TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.1
RANDOM_SEED = 42

# Prediction configuration
PREDICTION_HORIZON = 1  # Number of candles to predict ahead

# Logging configuration
LOG_LEVEL = 'INFO'

# API configuration (for future use)
API_HOST = '0.0.0.0'
API_PORT = 8000
API_DEBUG = False


def get_model_params(model_type: str) -> Dict[str, Any]:
    """
    Get default model parameters for the specified model type.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'lstm')
        
    Returns:
        Dictionary of model parameters
    """
    return MODEL_PARAMS.get(model_type.lower(), {})


def get_feature_config() -> Dict[str, Any]:
    """
    Get feature engineering configuration.
    
    Returns:
        Dictionary with feature configuration
    """
    return {
        'technical_indicators': DEFAULT_TECHNICAL_INDICATORS,
        'indicator_periods': INDICATOR_PERIODS,
        'lag_periods': DEFAULT_LAG_PERIODS,
    }


# Add more configuration functions as needed