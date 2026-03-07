import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from app.schemas import FeatureImpact
from typing import List
import json

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

    client = genai.Client(api_key=api_key)
    
    factor_strings = [f"- {f.feature}: {f.impact}" for f in top_factors]
    prompt = f"""
    Analyze this solar inverter risk assessment:
    Inverter ID: {inverter_id}
    Risk Probability: {risk_score:.2%}
    Risk Category: {risk_band}
    Top Technical Contributors:
    {chr(10).join(factor_strings)}

    Task:
    1. Write a 2-sentence 'narrative_summary' explaining the technical root cause based on the factors.
    2. Suggest 3 'recommended_actions' (max 7 words each).

    Output as JSON:
    {{
        "narrative_summary": "...",
        "recommended_actions": ["...", "...", "..."]
    }}
    """

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
        print(f"GenAI SDK Error: {e}")
        return f"System detected {risk_band} risk due to telemetry anomalies.", ["Inspect electrical connections", "Review recent maintenance logs"]
