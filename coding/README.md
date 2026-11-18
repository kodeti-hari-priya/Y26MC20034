# Financial Transaction Fraud Detection — GitHub + Vercel

This starter repository was prepared from the uploaded Colab notebook `P_trw.ipynb`.
The notebook implements financial-transaction fraud detection (target `is_fraud`: 0=Legitimate, 1=Fraud), compares five ML classifiers, and uses RandomOverSampler in the training pipeline. The deployment package uses the same core model families and selects the best one by fraud-class F1-score.

## Project structure

```text
fraud_detection_github_vercel/
├── api/
│   └── index.py                 # Vercel FastAPI prediction endpoint
├── artifacts/
│   ├── fraud_model.joblib       # Generated after training; add this to Git
│   └── model_schema.json        # Generated after training
├── data/
│   └── train_transactions.csv   # Put your dataset here (not included)
├── public/
│   └── index.html               # Web interface
├── train_model.py               # Trains and saves model + schema
├── requirements.txt
├── vercel.json
├── .gitignore
└── README.md
```

## Step 1 — Download your dataset from Colab

In Colab, locate the dataset you trained with (the CSV named `train_transactions.csv`). Download it to your computer. Do not upload private/customer-identifiable production data to a public GitHub repository. For a public project, use a suitably anonymized or synthetic dataset.

## Step 2 — Prepare the project on your computer

1. Extract this ZIP.
2. Open the extracted folder in VS Code.
3. Create a folder named `data`.
4. Copy `train_transactions.csv` into `data/`.
5. Open Terminal in VS Code at the project folder.

## Step 3 — Train the model locally

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
```

Successful training creates `artifacts/fraud_model.joblib` and `artifacts/model_schema.json`. The console prints the metrics and selected model. Check that both files exist before continuing.

**Important:** the training CSV must have a binary `is_fraud` column (0 and 1). The prediction page accepts amount, timestamp, customer location and payment method. Other columns present in your training dataset are filled by the model's fitted imputers at prediction time; this is a demonstration fallback, not a substitute for collecting all material predictors. If your model depends on additional fields, add them to the form and `api/index.py` before using it.

## Step 4 — Test the API locally

```bash
uvicorn api.index:app --reload
```

Open `http://127.0.0.1:8000/docs` to test `/api/predict`. The static webpage is served by Vercel; for a quick local page preview, use VS Code Live Server on `public/index.html` (the API must still be running).

## Step 5 — Push the project to GitHub

1. Sign in at https://github.com and create a **New repository**.
2. Give it a name such as `financial-transaction-fraud-detection`.
3. Choose Public or Private. Private is safer if your dataset or model contains sensitive information.
4. Do not initialize with a README if you are uploading this folder's README.
5. In VS Code, open Terminal in the project root and run:

```bash
git init
git add .
git status
git commit -m "Add fraud detection project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/financial-transaction-fraud-detection.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use your actual repository URL. If Git asks you to sign in, complete the browser authentication. Never put a GitHub password or access token in source code.

### If GitHub rejects the model file

GitHub blocks ordinary files larger than 100 MB. Check the size of `artifacts/fraud_model.joblib`. If it is over the limit, use Git LFS or store the model in an external model registry/object store and download it securely during deployment. Do not commit the raw dataset if it is large or private.

## Step 6 — Deploy on Vercel

1. Sign in at https://vercel.com using your GitHub account.
2. Click **Add New → Project**.
3. Import `financial-transaction-fraud-detection` from GitHub.
4. Keep the project root as the repository root. Vercel detects the FastAPI app in `api/index.py` automatically. The `public/` directory is served as static content.
5. Deploy.
6. Open the generated Vercel URL and test a transaction.

Vercel deploys from the GitHub repository. When you push later commits to the connected branch, Vercel normally creates a new deployment automatically.

## Step 7 — Verify the deployed project

- Open `https://YOUR-VERCEL-DOMAIN/api/health` — it should return `{"status":"ok"}`.
- Enter sample values that match the categories in your training dataset.
- Check both the classification and the probability values.
- If you see “Model files missing”, confirm that both files in `artifacts/` were generated and committed.
- If deployment fails due to model/package size or Python dependency compatibility, inspect the Vercel build log. Large ML models and scientific Python dependencies can exceed serverless size or execution limits; in that case host the model/API on Render or Hugging Face Spaces and keep the frontend on Vercel.

## Important project notes

1. The uploaded notebook uses `Random Forest`, `Logistic Regression`, `SVM`, `KNN`, and `Decision Tree`; this deployment training script evaluates the same five families.
2. The script uses a stratified 80/20 split, training-only random oversampling, preprocessing pipelines, and selects the highest fraud-class F1-score.
3. The notebook's SHAP/LIME explainability cells are research/analysis steps and are not included in this lightweight prediction API.
4. A model prediction is not a live banking integration. Do not connect it to real payment processing or automatically block transactions without security, privacy, fairness, and operational validation.
5. The dataset file is intentionally not bundled. Add your own CSV locally; avoid exposing real financial/customer data.

## Files generated after training

- `artifacts/fraud_model.joblib`: fitted preprocessing + oversampling + selected classifier pipeline.
- `artifacts/model_schema.json`: selected model name, feature names and evaluation metrics.
