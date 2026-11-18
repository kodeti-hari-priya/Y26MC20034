
from pathlib import Path
import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "fraud_model.joblib"
SCHEMA_PATH = ROOT / "artifacts" / "model_schema.json"

app = FastAPI(title="Financial Transaction Fraud Detection API")

# Enable frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaction(BaseModel):
    amount: float = Field(ge=0)
    timestamp: str
    customer_location: str
    payment_method: str

def load_assets():
    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        raise RuntimeError(
            "Model files missing. Run python train_model.py and include artifacts/."
        )

    return (
        joblib.load(MODEL_PATH),
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict")
def predict(tx: Transaction):
    try:
        model, schema = load_assets()

        stamp = pd.to_datetime(tx.timestamp, errors="coerce")

        if pd.isna(stamp):
            raise HTTPException(
                status_code=400,
                detail="Timestamp must be valid."
            )

        row = {
            "amount": tx.amount,
            "customer_location": tx.customer_location,
            "payment_method": tx.payment_method,
            "transaction_hour": stamp.hour,
            "transaction_day": stamp.day,
            "transaction_month": stamp.month,
            "transaction_weekday": stamp.dayofweek
        }

        frame = pd.DataFrame(
            [{col: row.get(col, None) for col in schema["feature_columns"]}],
            columns=schema["feature_columns"]
        )

        prediction = int(model.predict(frame)[0])
        probs = model.predict_proba(frame)[0]
        classes = list(model.classes_)

        fraud_prob = float(probs[classes.index(1)] * 100)
        legit_prob = float(probs[classes.index(0)] * 100)

        return {
            "prediction": "Fraud" if prediction == 1 else "Legitimate",
            "fraud_probability": round(fraud_prob, 2),
            "legitimate_probability": round(legit_prob, 2),
            "model": schema["model_name"]
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
