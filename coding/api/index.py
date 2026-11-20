
from pathlib import Path
import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "artifacts" / "fraud_model.joblib"
SCHEMA_PATH = ROOT / "artifacts" / "model_schema.json"


# --------------------------------------------------
# 2. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Financial Transaction Fraud Detection API",
    description="API for detecting fraudulent financial transactions using Machine Learning.",
    version="1.0.0"
)


# --------------------------------------------------
# 3. CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://y26-mc-20034.vercel.app",
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 4. TRANSACTION INPUT MODEL
# --------------------------------------------------

class Transaction(BaseModel):

    amount: float = Field(ge=0)

    timestamp: str

    customer_location: str

    payment_method: str


# --------------------------------------------------
# 5. LOAD TRAINED MODEL AND SCHEMA
# --------------------------------------------------

def load_assets():

    if not MODEL_PATH.exists():

        raise RuntimeError(
            "Trained model file is missing: artifacts/fraud_model.joblib"
        )

    if not SCHEMA_PATH.exists():

        raise RuntimeError(
            "Model schema file is missing: artifacts/model_schema.json"
        )

    model = joblib.load(MODEL_PATH)

    schema = json.loads(
        SCHEMA_PATH.read_text(encoding="utf-8")
    )

    return model, schema


# --------------------------------------------------
# 6. ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Financial Transaction Fraud Detection API",
        "status": "running",
        "health": "/api/health",
        "prediction": "/api/predict",
        "documentation": "/docs"
    }


# --------------------------------------------------
# 7. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "message": "Fraud Detection API is running successfully"
    }


# --------------------------------------------------
# 8. FRAUD PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/api/predict")
def predict(tx: Transaction):

    try:

        # Load trained model and schema
        model, schema = load_assets()

        # Convert timestamp
        stamp = pd.to_datetime(
            tx.timestamp,
            errors="coerce"
        )

        if pd.isna(stamp):

            raise HTTPException(
                status_code=400,
                detail="Timestamp must be valid."
            )

        # Prepare transaction features
        row = {

            "amount": tx.amount,

            "customer_location": tx.customer_location,

            "payment_method": tx.payment_method,

            "transaction_hour": stamp.hour,

            "transaction_day": stamp.day,

            "transaction_month": stamp.month,

            "transaction_weekday": stamp.dayofweek

        }

        # Match model training feature order
        feature_columns = schema["feature_columns"]

        frame = pd.DataFrame(
            [
                {
                    col: row.get(col, None)
                    for col in feature_columns
                }
            ],
            columns=feature_columns
        )

        # Generate prediction
        prediction = int(
            model.predict(frame)[0]
        )

        # Generate prediction probabilities
        if not hasattr(model, "predict_proba"):

            raise RuntimeError(
                "The trained model does not support predict_proba."
            )

        probs = model.predict_proba(frame)[0]

        classes = list(model.classes_)

        if 1 not in classes or 0 not in classes:

            raise RuntimeError(
                "Model must contain both class labels 0 and 1."
            )

        fraud_prob = float(
            probs[classes.index(1)] * 100
        )

        legit_prob = float(
            probs[classes.index(0)] * 100
        )

        # Return prediction result
        return {

            "prediction": (
                "Fraud" if prediction == 1 else "Legitimate"
            ),

            "fraud_probability": round(
                fraud_prob, 2
            ),

            "legitimate_probability": round(
                legit_prob, 2
            ),

            "model": schema["model_name"]

        }

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# --------------------------------------------------
# 9. RUN APPLICATION LOCALLY
# --------------------------------------------------

# Local command:
# uvicorn api.index:app --reload
#
# Render start command:
# uvicorn api.index:app --host 0.0.0.0 --port $PORT
