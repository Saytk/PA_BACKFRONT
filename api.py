from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.data_loader import load_csv
from src.features import (
    add_technical_indicators,
    add_temporal_features,
    add_lagged_features,
    add_return_features,
    add_custom_features,
)
from src.predictor import ModelPredictor
import pandas as pd

app = FastAPI()

# Autoriser les appels frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Peut être restreint à ["http://localhost:port"] en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CSV_PATH = "data/gold_data_last_90.csv"
TARGET_HORIZONS = [1, 2, 3]
FEATURE_COLUMNS = ['open', 'high', 'low', 'close']
LAGS = [1, 2, 3]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_technical_indicators(df)
    df = add_temporal_features(df)
    df = add_lagged_features(df, columns=FEATURE_COLUMNS, lags=LAGS)
    df = add_return_features(df)
    df = add_custom_features(df)
    df.ffill(inplace=True)  # Corrige les FutureWarning
    df.bfill(inplace=True)
    return df

@app.get("/")
def root():
    return {"message": "Quantia ML API is up."}

@app.get("/predict")
def get_latest_prediction():
    # Charger les données, générer les features et faire la prédiction
    df = load_csv(CSV_PATH)
    df = prepare_features(df)
    latest = df.iloc[[-1]]

    # Drop the date column before prediction as XGBoost doesn't support datetime type
    if 'date' in latest.columns:
        latest = latest.drop(columns=['date'])

    predictor = ModelPredictor(target_horizons=TARGET_HORIZONS)
    predictor.load_models()
    result = predictor.predict(latest)

    # Convert numpy.float32 values to Python native float types for JSON serialization
    serializable_result = {k: float(v) for k, v in result.items()}

    return serializable_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
