# Task 3 — Model Deployment with FastAPI

## Objective
Expose a trained classification model through a REST API.

## API
- `GET /` — health/status message
- `POST /predict` — prediction endpoint

## Setup
Copy a trained Task 1 model into:
`model/logistic_regression.joblib`

Then:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open the interactive API documentation:
`http://127.0.0.1:8000/docs`

## Request
The `/predict` endpoint expects all 30 Breast Cancer Wisconsin feature values.

See `sample_request.json`.
