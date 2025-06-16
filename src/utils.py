"""
Utility functions for the OHLC prediction project.

This module contains shared utility functions that can be used across the project.
"""

import os
import logging
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import from local modules
from src.config import LOG_LEVEL, MODELS_PATH, OHLC_COLUMNS


def setup_logging(log_level: str = LOG_LEVEL) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger instance
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger
    logger = logging.getLogger('ohlc_prediction')
    
    return logger


def generate_timestamp() -> str:
    """
    Generate a timestamp string for file naming.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def save_dict_to_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to the output JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_dict_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load a dictionary from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary loaded from the file
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics between true and predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    return metrics


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = 'Predictions vs Actual',
                    save_path: Optional[str] = None) -> None:
    """
    Plot predicted vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot (if None, display instead)
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    plt.plot(y_true, label='Actual', color='blue')
    
    # Plot predicted values
    plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_ohlc(df: pd.DataFrame, title: str = 'OHLC Chart', save_path: Optional[str] = None) -> None:
    """
    Plot OHLC data.
    
    Args:
        df: DataFrame containing OHLC data
        title: Plot title
        save_path: Path to save the plot (if None, display instead)
    """
    # TODO: Implement OHLC chart plotting
    # TODO: Consider using mplfinance for better OHLC visualization
    
    plt.figure(figsize=(12, 6))
    
    # Simple implementation - can be enhanced with candlestick charts
    plt.plot(df['close'], label='Close', color='black')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_experiment_dir(experiment_name: Optional[str] = None) -> str:
    """
    Create a directory for experiment results.
    
    Args:
        experiment_name: Name of the experiment (if None, use timestamp)
        
    Returns:
        Path to the created directory
    """
    if experiment_name is None:
        experiment_name = f"experiment_{generate_timestamp()}"
    
    experiment_dir = os.path.join(MODELS_PATH, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested dictionaries
        sep: Separator between keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":
    # Test utility functions
    # TODO: Add test code here
    pass