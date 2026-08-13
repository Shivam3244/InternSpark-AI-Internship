from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="InternSpark AI Classification API",
    version="1.0.0",
    description="REST API for a breast-cancer classification model."
)

MODEL_PATH = Path("model/logistic_regression.joblib")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=30, max_length=30)

@app.get("/")
def root():
    return {"message": "InternSpark AI API is running", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        return {"error": "Model not found. Copy logistic_regression.joblib into model/."}

    x = np.array(request.features, dtype=float).reshape(1, -1)
    prediction = int(model.predict(x)[0])
    probability = float(np.max(model.predict_proba(x)[0]))

    return {
        "prediction": prediction,
        "confidence": round(probability, 4)
    }
