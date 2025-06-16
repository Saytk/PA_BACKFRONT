import os
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class ModelTrainer:
    def __init__(
        self,
        model_dir: str = "models",
        target_horizons: List[int] = [1, 2, 3],
        model_type: str = "xgboost",
        model_params: Optional[Dict] = None,
    ):
        self.model_dir = model_dir
        self.target_horizons = target_horizons
        self.model_type = model_type.lower()
        self.model_params = model_params if model_params else {}
        self.models = {}

        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Train one model per prediction horizon (e.g., t+1, t+2, t+3).
        """
        for horizon in self.target_horizons:
            target_col = f"close_t+{horizon}"
            y_target = y[target_col]

            if self.model_type == "xgboost":
                model = xgb.XGBRegressor(**self.model_params)
                model.fit(X, y_target)
                self.models[f"t+{horizon}"] = model
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

    def evaluate(self, X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate each model using MAE metric.
        """
        metrics = {}
        for horizon in self.target_horizons:
            model = self.models.get(f"t+{horizon}")
            if model:
                preds = model.predict(X_val)
                true = y_val[f"close_t+{horizon}"]
                mae = np.mean(np.abs(true - preds))
                metrics[f"mae_t+{horizon}"] = mae
        return metrics

    def save_models(self):
        """
        Save trained models to disk.
        """
        for horizon, model in self.models.items():
            save_path = os.path.join(self.model_dir, f"model_{horizon}.json")  # ✅ correct
            model.save_model(save_path)
            print(f"✅ Model for {horizon} saved to: {save_path}")

    def load_models(self):
        """
        Load all models for the defined target horizons.
        """
        for horizon in self.target_horizons:
            model_path = os.path.join(self.model_dir, f"model_{horizon}.json")  # ✅ match trainer
            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                self.models[f"t+{horizon}"] = model
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
