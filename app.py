import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(
    title="TB Outcome Predictor API",
    description="Predicts TB treatment outcome (favorable/unfavorable) from SINAN notifications",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://geo-tb.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and features on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model_rs_0718.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "model", "features_rs_0718.pkl")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

print(f"Model loaded. Features {features}")


# Preprocessing helpers
def convert_age_to_years(nu_idade_n):
    if nu_idade_n is None or pd.isna(nu_idade_n):
        return np.nan
    nu_idade_n = int(nu_idade_n)
    unit = nu_idade_n // 1000
    value = nu_idade_n % 1000
    if unit == 1:
        return value / (24 * 365.25)  # hours
    elif unit == 2:
        return value / 365.25  # days
    elif unit == 3:
        return value / 12  # months
    elif unit == 4:
        return value  # years
    return np.nan


def idade_c_map(idade):
    if pd.isna(idade):
        return np.nan
    if idade < 19:
        return 0.0  # non-adult
    elif 19 <= idade <= 44:
        return 1.0  # adult
    elif 44 < idade <= 64:
        return 2.0  # middle-age
    elif idade >= 80:
        return 3.0  # 80+
    return np.nan


def cs_sexo_map(v):
    if v == "M":
        return 0.0
    elif v == "F":
        return 1.0
    return np.nan


def tratamento_map(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return np.nan
    v = str(v)
    if v == "1":
        return 0.0  # new case
    elif v == "2":
        return 1.0  # relapse
    elif v == "3":
        return 2.0  # re-entry
    return np.nan


def raiox_tora_map(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return np.nan
    v = str(v)
    if v == "1":
        return 0.0  # suspicious
    elif v == "2":
        return 1.0  # normal
    return np.nan


def preprocess(raw: dict) -> pd.DataFrame:
    """
    Apply the same preprocessing pipeline used during training.
    Accepts a raw SINAN notification dict, returns a single-row DataFrame
    with exactly the features the model expects.
    """
    df = pd.DataFrame([raw])

    # Age
    if "NU_IDADE_N" in df.columns:
        df["IDADE"] = df["NU_IDADE_N"].apply(convert_age_to_years).round(0)
    elif "IDADE" in df.columns:
        df["IDADE"] = pd.to_numeric(df["IDADE"], errors="coerce")
    else:
        df["IDADE"] = np.nan

    df["IDADE_c"] = df["IDADE"].apply(idade_c_map)

    # Gender
    if "CS_SEXO" in df.columns:
        df["CS_SEXO"] = df["CS_SEXO"].apply(
            lambda v: cs_sexo_map(str(v)) if pd.notna(v) else np.nan
        )

    # Treatment type
    if "TRATAMENTO" in df.columns:
        df["TRATAMENTO"] = df["TRATAMENTO"].apply(
            lambda v: tratamento_map(str(v)) if pd.notna(v) else np.nan
        )

    # Chest X-ray
    if "RAIOX_TORA" in df.columns:
        df["RAIOX_TORA"] = df["RAIOX_TORA"].apply(
            lambda v: raiox_tora_map(str(v)) if pd.notna(v) else np.nan
        )

    # Replace empty strings with NaN
    df = df.replace("", np.nan)

    # Comorbidities / vulnerability flags: '9' → NaN
    ignorado_cols = [
        "AGRAVAIDS",
        "AGRAVALCOO",
        "AGRAVDIABE",
        "AGRAVDOENC",
        "AGRAVOUTRA",
        "AGRAVDROGA",
        "AGRAVTABAC",
        "POP_LIBER",
        "POP_RUA",
        "POP_SAUDE",
        "POP_IMIG",
        "BENEF_GOV",
        "ANT_RETRO",
        "TRATSUP_AT",
        "CS_RACA",
        "CS_GESTANT",
        "INSTITUCIO",
        "DOENCA_TRA",
        "TRANSF",
        "RIFAMPICIN",
        "ISONIAZIDA",
        "ETAMBUTOL",
        "PIRAZINAMI",
        "ETIONAMIDA",
        "OUTRAS",
        "TRAT_SUPER",
    ]
    for col in ignorado_cols:
        if col in df.columns:
            df[col] = df[col].replace("9", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Bacteriology: '3' and '4' → NaN
    bac_cols = [
        "BACILOSC_E",
        "BACILOS_E2",
        "BACILOSC_O",
        "BACILOSC_1",
        "BACILOSC_2",
        "BACILOSC_3",
        "BACILOSC_4",
        "BACILOSC_5",
        "BACILOSC_6",
        "BAC_APOS_6",
    ]
    for col in bac_cols:
        if col in df.columns:
            df[col] = df[col].replace({"3": np.nan, "4": np.nan})
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # HIV: '3' and '4' → NaN
    if "HIV" in df.columns:
        df["HIV"] = df["HIV"].replace({"3": np.nan, "4": np.nan})
        df["HIV"] = pd.to_numeric(df["HIV"], errors="coerce")

    # HISTOPATOL: '4' and '5' → NaN
    if "HISTOPATOL" in df.columns:
        df["HISTOPATOL"] = df["HISTOPATOL"].replace({"4": np.nan, "5": np.nan})
        df["HISTOPATOL"] = pd.to_numeric(df["HISTOPATOL"], errors="coerce")

    # Culture: '3' and '4' → NaN
    for col in ["CULTURA_ES", "CULTURA_OU"]:
        if col in df.columns:
            df[col] = df[col].replace({"3": np.nan, "4": np.nan})
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Molecular test: '5' → NaN
    if "TEST_MOLEC" in df.columns:
        df["TEST_MOLEC"] = df["TEST_MOLEC"].replace("5", np.nan)
        df["TEST_MOLEC"] = pd.to_numeric(df["TEST_MOLEC"], errors="coerce")

    # Sensitivity test: '6' and '7' → NaN
    if "TEST_SENSI" in df.columns:
        df["TEST_SENSI"] = df["TEST_SENSI"].replace({"6": np.nan, "7": np.nan})
        df["TEST_SENSI"] = pd.to_numeric(df["TEST_SENSI"], errors="coerce")

    # Convert all remaining columns to numeric
    for col in df.columns:
        if col not in ["IDADE_c"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select only model features, add missing as NaN
    for feat in features:
        if feat not in df.columns:
            df[feat] = np.nan

    return df[features]


# Response model


class PredictionResponse(BaseModel):
    prediction: int  # 0 = favorable, 1 = unfavorable
    probability: float  # probability of unfavorable outcome
    risk_level: str  # LOW / MEDIUM / HIGH
    features_used: int  # how many features had non-null values


# Endpoints
@app.get("/")
def root():
    return {
        "service": "TB Outcome Predictor",
        "status": "running",
        "model_features": len(features),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request):
    try:
        raw = await request.json()
        X = preprocess(raw)

        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        features_used = int(X.notna().sum().sum())

        if probability < 0.35:
            risk_level = "LOW"
        elif probability < 0.65:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            features_used=features_used,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
def get_features():
    return {"features": features, "count": len(features)}
