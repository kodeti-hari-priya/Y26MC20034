"""
Train the fraud-detection models from train_transactions.csv.
Run: python train_model.py
Expected target column: is_fraud (0=legitimate, 1=fraud).
"""
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train_transactions.csv"
ARTIFACTS = ROOT / "artifacts"
TARGET = "is_fraud"
RANDOM_STATE = 42

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Put your CSV at data/train_transactions.csv."
        )
    df = pd.read_csv(DATA_PATH)
    if df.empty or TARGET not in df.columns:
        raise ValueError(f"CSV must be non-empty and contain '{TARGET}'.")
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("is_fraud must contain only 0 and 1, with no missing values.")

    df = df.drop_duplicates().dropna(subset=[TARGET]).copy()

    # Match the notebook's timestamp feature engineering.
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["transaction_hour"] = ts.dt.hour
        df["transaction_day"] = ts.dt.day
        df["transaction_month"] = ts.dt.month
        df["transaction_weekday"] = ts.dt.dayofweek
        df = df.drop(columns=["timestamp"])

    # Exclude record/customer identifiers from model inputs.
    id_cols = [
        c for c in df.columns
        if (c.lower() in {"transaction_id", "id", "uuid"} or c.lower().endswith("_id"))
        and c != TARGET
    ]
    X = df.drop(columns=[TARGET] + id_cols, errors="ignore")
    y = df[TARGET].astype(int)
    if X.empty:
        raise ValueError("No usable input features remain after removing identifiers.")
    if y.value_counts().min() < 2:
        raise ValueError("At least two rows from each target class are required.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                   random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced",
                   random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced",
                                                 random_state=RANDOM_STATE)
    }

    trained, rows = {}, []
    for name, estimator in candidates.items():
        pipe = ImbPipeline([
            ("preprocessor", clone(preprocessor)),
            ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
            ("classifier", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob))
        })
        trained[name] = pipe
        print(f"{name}: F1={rows[-1]['f1']:.4f}, recall={rows[-1]['recall']:.4f}")

    best = max(rows, key=lambda r: r["f1"])
    best_name = best["model"]
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(trained[best_name], ARTIFACTS / "fraud_model.joblib")
    schema = {
        "model_name": best_name,
        "feature_columns": list(X.columns),
        "input_columns": [c for c in ["amount", "customer_location", "payment_method",
                                       "transaction_hour", "transaction_day",
                                       "transaction_month", "transaction_weekday"]
                          if c in X.columns],
        "class_labels": {"0": "Legitimate", "1": "Fraud"},
        "metrics": rows,
        "note": "Other model features are left missing and handled by the fitted imputers."
    }
    (ARTIFACTS / "model_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"\nBest model: {best_name}")
    print(f"Saved model and schema under {ARTIFACTS}")

if __name__ == "__main__":
    main()
