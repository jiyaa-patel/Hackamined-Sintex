import json
from datetime import date
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Solar Inverter Risk Dashboard", layout="wide")

DEFAULT_API_BASE = "http://localhost:8000"


# -----------------------------
# Helpers
# -----------------------------
def call_health(api_base: str) -> bool:
    try:
        # Clean the base URL (strip trailing /predict if present)
        clean_base = api_base.rstrip("/").replace("/predict", "").replace("/chat", "")
        r = requests.get(f"{clean_base}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_predict(api_base: str, selected_date: date, inverter_id: int) -> Dict[str, Any]:
    # Clean the base URL (strip trailing /predict if present)
    clean_base = api_base.rstrip("/").replace("/predict", "").replace("/chat", "")
    # Match PredictHistoricalRequest: {inverter_id: str, timestamp: str}
    payload = {
        "inverter_id": str(inverter_id),
        "timestamp": f"{selected_date}T04:45:00.000Z"
    }
    r = requests.post(f"{clean_base}/predict", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def call_chat(api_base: str, query: str) -> Dict[str, Any]:
    # Clean the base URL (strip trailing /predict if present)
    clean_base = api_base.rstrip("/").replace("/predict", "").replace("/chat", "")
    # Match ChatRequest: {query: str, history: List[Dict]|None}
    payload = {
        "query": query,
        "history": []
    }
    r = requests.post(f"{clean_base}/chat", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def format_risk_band(band: str) -> str:
    b = str(band or "").lower().strip()
    if b in ("high", "h"):
        return "HIGH"
    if b in ("medium", "med", "m"):
        return "MEDIUM"
    if b in ("low", "l"):
        return "LOW"
    return str(band).upper()


def band_badge(band: str) -> str:
    b = format_risk_band(band)
    if b == "HIGH":
        return "🔴 HIGH"
    if b == "MEDIUM":
        return "🟠 MEDIUM"
    if b == "LOW":
        return "🟢 LOW"
    return b


def top_factors_to_df(top_factors: Any) -> pd.DataFrame:
    if not isinstance(top_factors, list):
        return pd.DataFrame(columns=["feature", "impact"])

    rows = []
    for item in top_factors:
        if not isinstance(item, dict):
            continue

        feature = item.get("feature") or item.get("name") or item.get("factor")
        impact = (
            item.get("impact")
            if item.get("impact") is not None
            else item.get("shap")
            if item.get("shap") is not None
            else item.get("value")
            if item.get("value") is not None
            else item.get("contribution")
        )

        if feature is None:
            continue

        try:
            impact_num = float(impact)
        except Exception:
            impact_num = None

        rows.append({"feature": str(feature), "impact": impact_num})

    out = pd.DataFrame(rows).dropna(subset=["impact"])
    if not out.empty:
        out = out.sort_values("impact", ascending=False).head(5)
    return out


# -----------------------------
# Title
# -----------------------------
st.title("AI-Driven Solar Inverter Risk Dashboard")


# -----------------------------
# Session state
# -----------------------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "selected_date" not in st.session_state:
    st.session_state.selected_date = date(2024, 3, 1)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Prediction Inputs")

    selected_date = st.date_input(
        "Input date",
        value=st.session_state.selected_date,
        min_value=date(2024, 3, 1),
        max_value=date(2024, 3, 26),
        help="Select the date for which you want prediction."
    )
    st.warning("Note: Historical data is only available for March 2024 (2024-03-01 to 2024-03-26).")

    selected_inverter_id = st.number_input(
        "Input inverter ID",
        min_value=1,
        max_value=12,
        value=1,
        step=1,
        help="Allowed inverter IDs: 1 to 12."
    )

    get_prediction = st.button("Get Prediction", use_container_width=True)

    st.divider()
    st.header("Backend")

    api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
    backend_ok = call_health(api_base)
    st.write("Backend status:", "✅ Online" if backend_ok else "⚠️ Offline")


# -----------------------------
# Predict call
# -----------------------------
if get_prediction:
    if not backend_ok:
        st.error("Backend is offline. Start the FastAPI service and try again.")
    else:
        try:
            result = call_predict(api_base, selected_date, selected_inverter_id)
            st.session_state.prediction_result = result
        except requests.HTTPError as e:
            try:
                err_json = e.response.json()
                st.error(f"API error: {err_json}")
            except Exception:
                st.error(f"API error: {e}")
        except Exception as e:
            st.error(f"Failed to call /predict: {e}")


# -----------------------------
# Main display
# -----------------------------
result = st.session_state.prediction_result

if result is None:
    st.info("Select a date and inverter ID, then click 'Get Prediction'.")
else:
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.subheader("Prediction Result")

        result_date = result.get("date", str(selected_date))
        result_inverter = result.get("inverter_id", selected_inverter_id)
        result_block = result.get("block", "N/A")
        result_score = result.get("risk_score", None)
        result_band = result.get("risk_band", "unknown")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Date", str(result_date))
        c2.metric("Inverter ID", str(result_inverter))
        c3.metric("Risk Score", f"{float(result_score):.3f}" if result_score is not None else "N/A")
        c4.metric("Risk Band", format_risk_band(result_band))

        st.markdown(f"**Block:** {result_block}")

        st.divider()
        st.subheader("AI Diagnostic Report")
        
        narrative = (
            result.get("narrative_summary")
            or result.get("summary")
            or result.get("narrative")
            or ""
        )
        if str(narrative).strip():
            st.info(str(narrative))
        else:
            st.warning("No narrative summary returned by backend.")

        recommendations = result.get("recommended_actions") or []
        if recommendations:
            st.markdown("**Recommended Actions:**")
            for rec in recommendations:
                st.markdown(f"- {rec}")

    with col_right:
        st.subheader("Top Contributing Factors")

        top_factors = (
            result.get("top_factors")
            or result.get("top_factors_json")
            or result.get("shap_top5")
            or []
        )

        factors_df = top_factors_to_df(top_factors)

        if factors_df.empty:
            st.info("No top factors returned by backend.")
        else:
            fig_bar = px.bar(
                factors_df,
                x="impact",
                y="feature",
                orientation="h",
                title="Top-5 Factors"
            )
            st.plotly_chart(fig_bar, use_container_width=True)




# -----------------------------
# Q&A Section
# -----------------------------
st.divider()
st.subheader("Operator Q&A (RAG)")

question = st.text_input(
    "Ask a question",
    value=""
)

colq1, colq2 = st.columns([0.2, 0.8])

with colq1:
    ask = st.button("Ask")

with colq2:
    st.caption("This calls your backend /chat endpoint.")

if ask:
    if not question.strip():
        st.warning("Type a question first.")
    elif not backend_ok:
        st.error("Backend is offline. Start the FastAPI service and try again.")
    else:
        try:
            res = call_chat(api_base, question.strip())
            answer = res.get("answer") or res.get("response") or str(res)

            st.write("### Answer")
            st.write(answer)

            sources = res.get("sources") or res.get("retrieved") or None
            if sources is not None:
                with st.expander("Show retrieved evidence (optional)"):
                    st.json(sources)

        except requests.HTTPError as e:
            try:
                err_json = e.response.json()
                st.error(f"API error: {err_json}")
            except Exception:
                st.error(f"API error: {e}")
        except Exception as e:
            st.error(f"Failed to call /chat: {e}")