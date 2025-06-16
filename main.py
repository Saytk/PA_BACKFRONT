import pandas as pd
from src.data_loader import load_csv, train_test_split_time_series
from src.features import (
    add_technical_indicators,
    add_temporal_features,
    add_lagged_features,
    add_return_features,
    add_custom_features,
)
from src.trainer import ModelTrainer
from src.predictor import ModelPredictor

# Configuration
CSV_PATH = "data/gold_data_last_90.csv"
TARGET_HORIZONS = [1, 2, 3]
FEATURE_COLUMNS = ['open', 'high', 'low', 'close']
LAGS = [1, 2, 3]
MODEL_PARAMS = {"n_estimators": 100, "max_depth": 3, "verbosity": 0}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_technical_indicators(df)
    df = add_temporal_features(df)
    df = add_lagged_features(df, columns=FEATURE_COLUMNS, lags=LAGS)
    df = add_return_features(df)
    df = add_custom_features(df)
    return df


def create_targets(df: pd.DataFrame, horizons: list) -> pd.DataFrame:
    for h in horizons:
        df[f"close_t+{h}"] = df["close"].shift(-h)
    return df


def main():
    # 1. Load and prepare data
    df = load_csv(CSV_PATH)
    df = prepare_features(df)
    df = create_targets(df, TARGET_HORIZONS)
    df.dropna(inplace=True)

    # 2. Split features and targets
    target_cols = [f"close_t+{h}" for h in TARGET_HORIZONS if f"close_t+{h}" in df.columns]
    X = df.drop(columns=target_cols + ['date'])  # ❗️Exclude 'date' to avoid XGBoost error
    y = df[target_cols]

    # 3. Train/test split
    X_train, X_val = train_test_split_time_series(X)
    y_train, y_val = train_test_split_time_series(y)

    # 4. Train
    trainer = ModelTrainer(target_horizons=TARGET_HORIZONS, model_params=MODEL_PARAMS)
    trainer.train(X_train, y_train)
    trainer.save_models()
    metrics = trainer.evaluate(X_val, y_val)
    print("📊 Evaluation metrics:", metrics)

    # 5. Predict on the latest row
    predictor = ModelPredictor(target_horizons=TARGET_HORIZONS)
    predictor.load_models()
    latest_features = X.iloc[[-1]]
    predictions = predictor.predict(latest_features)
    print("📈 Next predictions:", predictions)


if __name__ == "__main__":
    main()
