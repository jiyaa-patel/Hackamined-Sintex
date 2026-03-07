import sys
print("starting test...")

try:
    from app.services.genai_service import generate_risk_narrative
    from app.schemas import FeatureImpact
    print("imported!")
except Exception as e:
    print("Import error:", e)
    sys.exit(1)

try:
    print("calling GenAI...")
    factors = [FeatureImpact(feature="temp", impact=0.8)]
    res = generate_risk_narrative("1", 0.9, "HIGH", factors)
    print("result:", res)
except Exception as e:
    print("Execution error:", e)
print("Finished!")
