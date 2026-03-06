import os, json, joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import shap

CSV_PATH = os.getenv("CSV_PATH", "part_001.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

SAMPLE_INTERVAL_MIN = 5

app = FastAPI(title="Inverter Risk API", version="1.0")

raw_df = None
long_df = None
bundle = None
pipe = None
schema = None
explainer = None

class PredictRequest(BaseModel):
    mac: str
    inverter_index: int = Field(..., ge=0, le=200)
    target_date: str = Field(..., examples=["2024-03-15"])

class PredictResponse(BaseModel):
    mac: str
    inverter_index: int
    target_date: str
    risk_score: float
    prediction: int
    top_factors: list

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    inv_cols = [c for c in df.columns if c.startswith("inverters[")]
    inv_idxs = sorted(set(int(c.split("[")[1].split("]")[0]) for c in inv_cols))

    frames = []
    for i in inv_idxs:
        sub_cols = ["mac", "timestamp", "dt", "sensors[0].ambient_temp"]
        mapping = {"sensors[0].ambient_temp": "ambient_temp"}

        for name, col in {
            "pv1_power": f"inverters[{i}].pv1_power",
            "op_state": f"inverters[{i}].op_state",
            "temp": f"inverters[{i}].temp",
            "v_ab": f"inverters[{i}].v_ab",
            "v_bc": f"inverters[{i}].v_bc",
            "v_ca": f"inverters[{i}].v_ca",
        }.items():
            if col in df.columns:
                sub_cols.append(col)
                mapping[col] = name

        sub = df[sub_cols].rename(columns=mapping)
        sub["inverter_index"] = i
        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)
    return out

def compute_features(window: pd.DataFrame) -> dict:
    p = window["pv1_power"].astype(float)
    op = window["op_state"].astype(float)
    temp = window.get("temp", pd.Series([np.nan]*len(window))).astype(float)
    amb = window.get("ambient_temp", pd.Series([np.nan]*len(window))).astype(float)

    up = ((op == -1) | (p > 1.0)).astype(int)
    shut = ((op == 0) & (p <= 1.0)).astype(int)

    shutdown_minutes = int(shut.sum() * SAMPLE_INTERVAL_MIN)

    lengths = []
    cur = 0
    for v in shut.values:
        if v == 1: cur += 1
        else:
            if cur > 0: lengths.append(cur)
            cur = 0
    if cur > 0: lengths.append(cur)
    shutdown_events = int(sum(1 for L in lengths if L >= 12))  # >= 60 mins

    x = np.arange(len(p))
    if np.isfinite(p).sum() >= 2:
        mask = np.isfinite(p.values)
        slope = float(np.polyfit(x[mask], p.values[mask], 1)[0])
    else:
        slope = np.nan

    v_cols = [c for c in ["v_ab", "v_bc", "v_ca"] if c in window.columns]
    if v_cols:
        v_std = window[v_cols].astype(float).std(skipna=True)
        vline_std_mean = float(np.nanmean(v_std.values))
    else:
        vline_std_mean = np.nan

    return {
        "power_mean_30d": float(np.nanmean(p)),
        "power_std_30d": float(np.nanstd(p)),
        "power_zero_ratio_30d": float(np.nanmean((p <= 1.0).astype(int))),
        "power_trend_30d": slope,
        "uptime_ratio_30d": float(np.nanmean(up)),
        "shutdown_events_30d": shutdown_events,
        "shutdown_minutes_30d": shutdown_minutes,
        "temp_mean_30d": float(np.nanmean(temp)),
        "temp_std_30d": float(np.nanstd(temp)),
        "vline_std_mean_30d": vline_std_mean,
        "ambient_temp_mean_30d": float(np.nanmean(amb)),
    }

@app.on_event("startup")
def startup():
    global raw_df, long_df, bundle, pipe, schema, explainer

    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"CSV not found: {CSV_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH} (run train.py first)")

    raw_df = pd.read_csv(CSV_PATH)
    long_df = wide_to_long(raw_df).dropna(subset=["pv1_power", "op_state"], how="all")
    long_df = long_df.sort_values(["mac", "inverter_index", "dt"]).reset_index(drop=True)

    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    schema = bundle["schema"]

    # For RF, SHAP explainer is for the underlying tree model
    explainer = shap.TreeExplainer(pipe.named_steps["model"])

@app.get("/health")
def health():
    return {"status": "ok", "csv_loaded": long_df is not None, "model_loaded": pipe is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    target = pd.Timestamp(req.target_date).tz_localize("UTC")
    start = target - pd.Timedelta(days=schema["history_days"])

    g = long_df[(long_df["mac"] == req.mac) & (long_df["inverter_index"] == req.inverter_index)]
    if g.empty:
        raise HTTPException(404, detail={"error": "NotFound", "message": "mac/inverter_index not found"})

    window = g[(g["dt"] >= start) & (g["dt"] < target)]
    if len(window) < (24*60//SAMPLE_INTERVAL_MIN):  # need >=1 day of samples
        raise HTTPException(422, detail={
            "error": "ValidationError",
            "message": "Not enough history in last 30 days for this inverter/date"
        })

    feats = compute_features(window)
    X = pd.DataFrame([feats])[schema["required_features"]]

    # predict
    risk = float(pipe.predict_proba(X)[:, 1][0])
    pred = int(risk >= 0.5)

    # SHAP top-5
    X_imp = pipe.named_steps["imputer"].transform(X)
    shap_vals = explainer.shap_values(X_imp)[1][0]
    names = schema["required_features"]

    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
    top = [{"feature": names[i], "value": float(X.iloc[0, i]), "shap": float(shap_vals[i])} for i in top_idx]

    return PredictResponse(
        mac=req.mac,
        inverter_index=req.inverter_index,
        target_date=req.target_date,
        risk_score=risk,
        prediction=pred,
        top_factors=top
    )