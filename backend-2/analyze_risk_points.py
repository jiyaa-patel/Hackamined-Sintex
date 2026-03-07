import joblib
import pandas as pd
import json
import os
import csv
import io
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "app/services/model.pkl")
SCHEMA_PATH = os.path.join(BASE_DIR, "app/services/feature_schema.json")
CSV_PATH = os.path.join(BASE_DIR, "data/historical_telemetry.csv")

# Load model and schema
model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH, "r") as f:
    FEATURE_ORDER = json.load(f).get("features", [])

def get_risk(inverter_id, features):
    ordered_values = [features.get(f, 0.0) for f in FEATURE_ORDER]
    X = pd.DataFrame([ordered_values], columns=FEATURE_ORDER)
    risk_score = float(model.predict_proba(X)[0][1])
    
    # Apply variance logic from ml_service.py
    try:
        inv_id_num = int(inverter_id)
        variance = (inv_id_num * 0.037) % 0.12
        if risk_score > 0.8:
            risk_score = max(0.01, risk_score - variance)
        else:
            risk_score = min(0.99, risk_score + variance)
    except:
        pass
        
    band = "LOW"
    if risk_score > 0.7: band = "HIGH"
    elif risk_score > 0.5: band = "MEDIUM"
    
    return risk_score, band

print("Scanning historical_telemetry.csv for non-LOW risk points...")
high_count = 0
medium_count = 0
total_checked = 0

# Scan first 5000 rows as a sample
with open(CSV_PATH, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        total_checked += 1
        if total_checked > 5000: break
        
        # We need to test for each inverter 1-12 as data_service does
        for i in range(1, 13):
            inverter_id = str(i)
            # mapping logic from data_service.py
            idx_in_row = 123 + (i-1)
            raw_id = row[idx_in_row].strip().split('.')[0]
            if raw_id == inverter_id:
                raw_power = float(row[111 + (i-1)]) if row[111 + (i-1)] else 0.0
                raw_temp = float(row[63 + (i-1)]) if row[63 + (i-1)] else 0.0
                raw_freq = float(row[75 + (i-1)]) if row[75 + (i-1)] else 0.0
                raw_v_ab = float(row[39 + (i-1)]) if row[39 + (i-1)] else 0.0
                raw_v_bc = float(row[27 + (i-1)]) if row[27 + (i-1)] else 0.0
                raw_v_ca = float(row[15 + (i-1)]) if row[15 + (i-1)] else 0.0
                
                base_variance = (int(inverter_id) * 11) % 7
                temp_modifier = base_variance * 1.5 
                voltage_modifier = base_variance * 3.0
                
                features = {
                    "power_mean_30d": raw_power + (base_variance * 0.5),
                    "power_std_30d": max(0.1, (raw_power * 0.1) + (base_variance * 0.05)),
                    "temp_mean_30d": max(20.0 + temp_modifier, raw_temp + temp_modifier),
                    "temp_std_30d": max(0.5, (raw_temp * 0.05) + (base_variance * 0.1)),
                    "freq_mean_30d": 50.0 + (base_variance * 0.02) if raw_freq == 0.0 else raw_freq,
                    "freq_std_30d": 0.1 + (base_variance * 0.01),
                    "v_ab_mean_30d": max(220.0 + voltage_modifier, raw_v_ab + voltage_modifier),
                    "v_ab_std_30d": 5.0 + (base_variance * 0.5),
                    "v_bc_mean_30d": max(220.0 + voltage_modifier, raw_v_bc + voltage_modifier),
                    "v_bc_std_30d": 5.0 + (base_variance * 0.5),
                    "v_ca_mean_30d": max(220.0 + voltage_modifier, raw_v_ca + voltage_modifier),
                    "v_ca_std_30d": 5.0 + (base_variance * 0.5),
                    "ambient_temp": 30.0 + base_variance,
                    "voltage_imbalance": max(0.0, abs(raw_v_ab - raw_v_bc) + (base_variance * 0.5))
                }
                
                score, band = get_risk(inverter_id, features)
                if band == "HIGH":
                    high_count += 1
                    # Find timestampDate at index 437 (0-indexed)
                    ts_date = row[437] if len(row) > 437 else "Unknown"
                    print(f"HIGH risk found: Inv {inverter_id}, Row {total_checked}, Date {ts_date}, Score {score:.4f}")
                elif band == "MEDIUM":
                    medium_count += 1
                    # print(f"MEDIUM risk found: Inv {inverter_id}, Row {total_checked}, Score {score:.4f}")

print(f"\nScan complete. Checked {total_checked} rows.")
print(f"Found {high_count} HIGH risk points and {medium_count} MEDIUM risk points.")
