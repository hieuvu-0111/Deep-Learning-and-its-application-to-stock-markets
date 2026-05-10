import tensorflow as tf
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# Load model once at startup
model = tf.keras.models.load_model("../models/task2_3_vietnam_model.keras")
print("Model loaded successfully")

WINDOW_SIZE  = 30
NUM_FEATURES = 5    # Open, High, Low, Close, Volume
K            = 7    # forecast horizon (days)

app = FastAPI(
    title="Stock Price Prediction API",
    description="Vietnam stock K-day price forecast using CNN (Task 2.3)",
    version="1.0.0"
)

# Request schema
class PredictRequest(BaseModel):
    ticker:    str = "HPG"
    instances: list   # shape: (30, 5) - 30 days × OHLCV

    @field_validator("instances")
    @classmethod
    def check_shape(cls, v):
        if len(v) != WINDOW_SIZE:
            raise ValueError(
                f"Expected {WINDOW_SIZE} rows, got {len(v)}"
            )
        for row in v:
            if len(row) != NUM_FEATURES:
                raise ValueError(
                    f"Each row must have {NUM_FEATURES} values "
                    f"(Open, High, Low, Close, Volume)"
                )
        return v

# Response schema 
class PredictResponse(BaseModel):
    ticker:     str
    prediction: list   # shape: (K,) - K predicted closing prices
    horizon:    int

# Endpoints
@app.get("/")
def root():
    return {
        "message": "Stock Price Prediction API is running",
        "docs":    "http://127.0.0.1:8000/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "task2_3_vietnam_model.keras"}

@app.post("/predict", response_model=PredictResponse)
def predict_price(request: PredictRequest):
    try:
        # Convert to numpy and normalize per window
        X = np.array(request.instances, dtype=float)  # (30, 5)

        # Per-window min-max normalization - same as training
        f_min  = X.min(axis=0)
        f_max  = X.max(axis=0)
        f_diff = np.where(f_max - f_min == 0, 1e-8, f_max - f_min)
        X_norm = (X - f_min) / f_diff

        # Add batch dimension -> (1, 30, 5)
        X_input = X_norm[np.newaxis, :, :]

        # Predict normalized output -> (1, K)
        y_pred_norm = model.predict(X_input, verbose=0)

        # Denormalize using Close column (index 3 in OHLCV)
        close_idx   = 3
        y_pred_denorm = (y_pred_norm[0]
                         * f_diff[close_idx]
                         + f_min[close_idx])

        return PredictResponse(
            ticker=request.ticker,
            prediction=y_pred_denorm.tolist(),
            horizon=K
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))