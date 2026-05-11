# TB Outcome Predictor API

FastAPI service for predicting TB treatment outcomes from SINAN notifications.
Trained on RS 2007-2018 data using XGBoost with optimized hyperparameters.

## Model

- **Algorithm**: XGBoost (optimized via RandomizedSearchCV)
- **Training data**: Rio Grande do Sul, 2007-2018 (~45k records)
- **Target**: Binary outcome — 0 (favorable: cure) / 1 (unfavorable: dropout or death)
- **Metrics**: Sensitivity 0.8434 · AUC-ROC 0.9492 · F1-macro 0.8509

## Setup

### Local development

```bash
pip install -r requirements.txt

# Place your model files in the model/ directory:
# model/xgb_final.pkl
# model/features_final.pkl

uvicorn app:app --reload --port 8000
```

### Railway deployment

1. Push this repo to GitHub
2. Create a new Railway project from the GitHub repo
3. Add model files to `model/` directory
4. Railway auto-deploys on push

## API Endpoints

### `GET /`
Health check and service info.

### `GET /health`
Health check for Railway.

### `POST /predict`
Predict TB treatment outcome for a single notification.

**Request body** — raw SINAN notification fields (all optional):
```json
{
  "NU_IDADE_N": "4030",
  "CS_SEXO": "M",
  "TRATAMENTO": "1",
  "HIV": "2",
  "AGRAVDROGA": "2",
  "POP_RUA": "2",
  ...
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.7823,
  "risk_level": "HIGH",
  "features_used": 14
}
```

- `prediction`: 0 = favorable, 1 = unfavorable
- `probability`: probability of unfavorable outcome (0-1)
- `risk_level`: LOW (<0.35) / MEDIUM (0.35-0.65) / HIGH (>0.65)
- `features_used`: number of non-null features the model used

### `GET /features`
Returns the list of features the model expects.

## Next.js Integration

```typescript
// lib/predict.ts
export async function predictOutcome(notification: Record<string, string>) {
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(notification),
  })

  if (!response.ok) throw new Error('Prediction failed')
  return response.json()
}
```

## Project Structure

```
tb-predictor-api/
├── app.py              # FastAPI application + preprocessing pipeline
├── requirements.txt    # Python dependencies
├── railway.json        # Railway deployment config
├── Procfile            # Process definition
├── model/
│   ├── xgb_final.pkl       # Trained XGBoost model
│   └── features_final.pkl  # Feature list
└── README.md
```