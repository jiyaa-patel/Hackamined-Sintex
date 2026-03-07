# Hackamined-Sintex

## 🚀 AI-Driven Solar Inverter Risk Dashboard

An advanced predictive maintenance platform that leverages Machine Learning to anticipate solar inverter failures before they occur. By analyzing high-frequency telemetry data, the system identifies anomalies like voltage imbalances, thermal stress, and power fluctuations, providing actionable AI-generated diagnostic reports.

### ✨ Key Features
- **Predictive Risk Scoring**: Real-time failure probability (0-100%) for 12+ solar inverters.
- **AI Diagnostics**: Local LLM-powered (Ollama/Llama3) narrative reports explaining *why* an inverter is at risk.
- **Historical Analysis**: Exploratory dashboard covering telemetry from March 2024 to March 2026.
- **Dynamic Risk Visualization**: Color-coded risk bands (HIGH/MEDIUM/LOW) with feature-level contribution analysis (SHAP).
- **5-Minute Telemetry Precision**: High-resolution time-series data analysis.

### 🛠️ Tech Stack
- **Frontend**: Streamlit, Plotly, Pandas
- **Backend**: FastAPI, Scikit-Learn, Joblib
- **AI/LLM**: Ollama (Llama 3), RAG (Retrieval-Augmented Generation)
- **Database**: CSV-based high-performance telemetry indexing

### ⚙️ Setup & Installation
1. **Backend**:
   ```bash
   cd backend-2
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
2. **Frontend**:
   ```bash
   cd frontend
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. **Local AI**:
   Install Ollama and run `ollama pull llama3`.

---

### 👥 Team Members
1. Shubham (@shubham280706)
2. Jiya (@jiyaa-patel)
3. Megh (@Megh36)
4. Shrey


team members:
1. Shubham (shubham280706)
2. Jiya (jiyaa-patel)
3. Megh (Megh36)
4. Shrey (Shrey_9075)

