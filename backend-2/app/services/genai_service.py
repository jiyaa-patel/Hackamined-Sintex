import os
import json
import ollama
from dotenv import load_dotenv
from app.schemas import FeatureImpact
from typing import List

def generate_risk_narrative(inverter_id: str, risk_score: float, risk_band: str, top_factors: List[FeatureImpact]):
    """
    Constructs a diagnostic AI narrative using local Ollama.
    """
    load_dotenv(override=True)
    model = os.environ.get("OLLAMA_MODEL", "llama3")
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    # Initialize client (implicit in simplified ollama lib, but setting host via env is standard)
    # The ollama-python library uses OLLAMA_HOST env var by default.

    factor_strings = [f"- {f.feature}: {f.impact}" for f in top_factors]
    prompt = f"""
    Analyze solar inverter risk:
    Inverter: {inverter_id}, Risk: {risk_score:.2%} ({risk_band})
    Factors: {', '.join(factor_strings)}

    Output JSON Format:
    {{
        "narrative_summary": "2-sentence technical root cause summary.",
        "recommended_actions": ["Action 1", "Action 2", "Action 3"]
    }}
    """

    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            format='json'
        )
        
        data = json.loads(response['message']['content'])
        return data.get("narrative_summary"), data.get("recommended_actions")
        
    except Exception as e:
        print(f"Ollama Error: {e}")
        # Local fallback if Ollama fails
        return f"System detected {risk_band} risk due to technical anomalies in inverter telemetry.", \
               ["Inspect electrical connections", "Check cooling systems", "Review maintenance logs"]

