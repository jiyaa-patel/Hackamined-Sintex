import os
import sys
from dotenv import load_dotenv
from app.services.genai_service import generate_risk_narrative
from app.schemas import FeatureImpact

load_dotenv()

def test_genai():
    print("Testing GenAI Service...")
    print(f"API Key: {os.environ.get('GEMINI_API_KEY')[:10]}...")
    
    factors = [
        FeatureImpact(feature="temperature", impact=0.8),
        FeatureImpact(feature="voltage", impact=0.2)
    ]
    
    try:
        # Manually calling with 1.5-flash to see if quota is different
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents="Say hello"
        )
        print(f"Direct 1.5-flash test: {response.text}")
        
        narrative, actions = generate_risk_narrative(
            inverter_id="INV-001",
            risk_score=0.15,
            risk_band="LOW",
            top_factors=factors
        )
        print("\n--- RESULTS ---")
        print(f"Narrative: {narrative}")
        print(f"Actions: {actions}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_genai()
