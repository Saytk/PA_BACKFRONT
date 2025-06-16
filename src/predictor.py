import os
import xgboost as xgb
import pandas as pd
from typing import List, Dict


class ModelPredictor:
    def __init__(self, model_dir: str = "models", target_horizons: List[int] = [1, 2, 3]):
        """
        Initialize the predictor with the directory containing saved models.
        """
        self.model_dir = model_dir
        self.target_horizons = target_horizons
        self.models = {}

    def load_models(self):
        """
        Load all models for the defined target horizons.
        """
        for horizon in self.target_horizons:
            model_path = os.path.join(self.model_dir, f"model_t+{horizon}.json")
            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                self.models[f"t+{horizon}"] = model
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

    def predict(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Make predictions for each horizon.

        Args:
            X: Input DataFrame with features (should be a single row)

        Returns:
            Dictionary with predictions per horizon.
        """
        if not self.models:
            raise RuntimeError("Models not loaded. Call `load_models()` first.")

        if len(X) != 1:
            raise ValueError("Input X must contain exactly one row for prediction.")

        predictions = {}
        for horizon, model in self.models.items():
            pred = model.predict(X)[0]
            predictions[f"close_{horizon}"] = pred

        return predictions
