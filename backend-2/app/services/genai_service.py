import os
import time
import random
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from app.schemas import FeatureImpact
from typing import List

def generate_risk_narrative(inverter_id: str, risk_score: float, risk_band: str, top_factors: List[FeatureImpact]):
    """
    Constructs a diagnostic AI narrative using the modern Google GenAI SDK.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        api_key = api_key.strip('"').strip("'")
    
    if not api_key or not api_key.startswith("AIza"):
        return f"Operational analysis indicates a {risk_band} risk level.", ["Check system logs", "Monitor performance trends"]

    client = genai.Client(api_key=api_key, http_options={'timeout': 10.0})
    
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

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-flash-latest',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json'
                )
            )
            
            data = json.loads(response.text)
            return data.get("narrative_summary"), data.get("recommended_actions")
            
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "rate" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.random()
                    print(f"GenAI Rate Limited (Attempt {attempt+1}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
            print(f"GenAI SDK Error: {e}")
            break

    return f"System detected {risk_band} risk due to telemetry anomalies.", ["Inspect electrical connections", "Review recent maintenance logs"]
