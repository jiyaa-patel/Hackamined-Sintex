# train.py
import os, json, joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import shap

CSV_PATH = os.getenv("CSV_PATH", r"data\raw\part_001.csv")
SCHEMA_PATH = "feature_schema.json"
MODEL_OUT = os.getenv("MODEL_OUT", r"models\model.joblib")

SAMPLE_INTERVAL_MIN = 5  # dataset looks like 5-min telemetry

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    inv_cols = [c for c in df.columns if c.startswith("inverters[")]
    inv_idxs = sorted(set(int(c.split("[")[1].split("]")[0]) for c in inv_cols))

    frames = []
    for i in inv_idxs:
        base = ["mac", "timestamp", "dt"]
        mapping = {}

        # ambient temp
        if "sensors[0].ambient_temp" in df.columns:
            base.append("sensors[0].ambient_temp")
            mapping["sensors[0].ambient_temp"] = "ambient_temp"

        wanted = {
            "pv1_power": f"inverters[{i}].pv1_power",
            "op_state": f"inverters[{i}].op_state",
            "temp": f"inverters[{i}].temp",
            "v_ab": f"inverters[{i}].v_ab",
            "v_bc": f"inverters[{i}].v_bc",
            "v_ca": f"inverters[{i}].v_ca",
        }

        for name, col in wanted.items():
            if col in df.columns:
                base.append(col)
                mapping[col] = name

        sub = df[base].rename(columns=mapping)
        sub["inverter_index"] = i
        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)
    return out

def compute_features(window: pd.DataFrame) -> dict:
    p = window["pv1_power"].astype(float)
    op = window["op_state"].astype(float)
    temp = window["temp"].astype(float) if "temp" in window.columns else pd.Series([np.nan]*len(window))
    amb = window["ambient_temp"].astype(float) if "ambient_temp" in window.columns else pd.Series([np.nan]*len(window))

    up = ((op == -1) | (p > 1.0)).astype(int)
    shut = ((op == 0) & (p <= 1.0)).astype(int)

    shutdown_minutes = int(shut.sum() * SAMPLE_INTERVAL_MIN)

    # count shutdown segments >= 60 minutes (12 samples)
    lengths = []
    cur = 0
    for v in shut.values:
        if v == 1:
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
            cur = 0
    if cur > 0:
        lengths.append(cur)
    shutdown_events = int(sum(1 for L in lengths if L >= 12))

    # linear slope for power trend
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

def future_failure_label(future_window: pd.DataFrame) -> int:
    if future_window.empty:
        return 0
    p = future_window["pv1_power"].astype(float)
    op = future_window["op_state"].astype(float)
    shut = ((op == 0) & (p <= 1.0)).astype(int)
    shutdown_minutes = int(shut.sum() * SAMPLE_INTERVAL_MIN)
    return int(shutdown_minutes >= 60)  # >= 1 hour in next 7 days

def evaluate(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc)}

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH} (put it in data/raw/part_001.csv)")

    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    raw = pd.read_csv(CSV_PATH)
    long = wide_to_long(raw).dropna(subset=["pv1_power", "op_state"], how="all")
    long = long.sort_values(["mac", "inverter_index", "dt"]).reset_index(drop=True)

    hist_days = schema.get("history_days", 30)
    label_days = schema.get("label_window_days", 7)

    samples = []

    for (mac, inv), g in long.groupby(["mac", "inverter_index"]):
        g = g.sort_values("dt")
        g["day"] = g["dt"].dt.floor("D")
        days = sorted(g["day"].unique())

        for d in days:
            end = pd.Timestamp(d).tz_localize("UTC")  # day boundary
            start = end - pd.Timedelta(days=hist_days)
            fut_end = end + pd.Timedelta(days=label_days)

            hist = g[(g["dt"] >= start) & (g["dt"] < end)]
            fut = g[(g["dt"] >= end) & (g["dt"] < fut_end)]

            # need at least 1 day history samples
            if len(hist) < (24 * 60 // SAMPLE_INTERVAL_MIN):
                continue

            feats = compute_features(hist)
            y = future_failure_label(fut)

            row = {"mac": mac, "inverter_index": int(inv), "day": str(end.date()), "label": y}
            row.update(feats)
            samples.append(row)

    feat_df = pd.DataFrame(samples).sort_values(["mac", "inverter_index", "day"])
    print("Training samples:", feat_df.shape)
    if feat_df.empty:
        raise RuntimeError("No training samples created. Check timestamps / mac / inverter columns.")

    X = feat_df[schema["required_features"]]
    y = feat_df["label"].astype(int).values

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ))
    ])

    # Time split CV
    tscv = TimeSeriesSplit(n_splits=5)
    ms = []
    for fold, (tr, va) in enumerate(tscv.split(X), start=1):
        pipe.fit(X.iloc[tr], y[tr])
        prob = pipe.predict_proba(X.iloc[va])[:, 1]
        m = evaluate(y[va], prob)
        m["fold"] = fold
        ms.append(m)

    mdf = pd.DataFrame(ms)
    print("\nCV metrics:\n", mdf)
    print("\nCV mean:\n", mdf.drop(columns=["fold"]).mean(numeric_only=True))

    # Fit final
    pipe.fit(X, y)

    # SHAP explainer for RF model
    explainer = shap.TreeExplainer(pipe.named_steps["model"])

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "schema": schema, "feature_names": schema["required_features"]},
        MODEL_OUT
    )
    print(f"\nSaved model: {MODEL_OUT}")

if __name__ == "__main__":
    main()