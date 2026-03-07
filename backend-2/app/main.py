from fastapi import FastAPI, HTTPException, Body
from app.schemas import PredictRequest, PredictResponse, PredictHistoricalRequest, ChatRequest, ChatResponse
import csv
import io
from app.services.ml_service import predict_risk
from app.services.genai_service import generate_risk_narrative
from app.services.data_service import get_historical_features
from app.services.rag_service import get_chat_recommendations


app = FastAPI(
    title="Inverter Risk Processing API",
    description="A sequential pipeline moving from telemetry features to ML inference to GenAI summaries."
)

@app.post("/chat", response_model=ChatResponse)
def technical_expert_chat(request: ChatRequest):
    """
    Expert RAG Chatbot focused on solar inverter maintenance.
    Consults the technical manual context to provide accurate recommendations.
    """
    try:
        recommendation = get_chat_recommendations(request.query, request.history)
        return ChatResponse(answer=recommendation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Backend-2 is running effectively."}

@app.post("/predict", response_model=PredictResponse)
def analyze_inverter_risk(request: PredictRequest):
    """
    1. Receive structured JSON payload with precomputed features.
    2. Pass to ML Model for dynamic risk_score and risk_band.
    3. Generate GenAI narrative summary & recommendations.
    4. Compile final Payload.
    """
    try:
        # Step 2: Extract ML metrics
        risk_score, risk_band, top_factors = predict_risk(request)
        
        # Step 3: Trigger GenAI Analysis
        narrative, recommendations = generate_risk_narrative(
            request.inverter_id,
            risk_score,
            risk_band,
            top_factors
        )
        
        # Step 4: Construct Output
        return PredictResponse(
            inverter_id=request.inverter_id,
            block=request.block,
            risk_score=risk_score,
            risk_band=risk_band,
            top_factors=top_factors,
            narrative_summary=narrative,
            recommended_actions=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-csv", response_model=list[PredictResponse])
def analyze_csv_batch(csv_text: str = Body(..., media_type="text/plain")):
    """
    Expects a standard CSV where headers match the InferenceRequest schema.
    Returns an array of standard PredictResponses.
    """
    try:
        f = io.StringIO(csv_text.strip())
        reader = csv.DictReader(f)
        
        results = []
        for row in reader:
            inverter_id = row.pop("inverter_id", "Unknown")
            block = row.pop("block", "Unknown")
            timestamp = row.pop("timestamp", None)
            
            # All remaining fields are parsed as floats into the features dict
            features = {k: float(v) for k, v in row.items() if v.strip()}
            
            req = PredictRequest(
                inverter_id=inverter_id,
                block=block,
                timestamp=timestamp,
                features=features
            )
            
            # Predict and append
            results.append(analyze_inverter_risk(req))
            
        return results
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

@app.post("/predict-historical", response_model=PredictResponse)
def analyze_historical_risk(request: PredictHistoricalRequest):
    """
    Looks up pre-recorded raw telemetry from the 455MB CSV database based on 
    Inverter ID and Timestamp, compiles the ML features, and predicts risk.
    """
    try:
        features = get_historical_features(request.inverter_id, request.timestamp)
        
        if not features:
            raise HTTPException(
                status_code=404, 
                detail=f"No telemetry found for Inverter '{request.inverter_id}' near time '{request.timestamp}'"
            )
            
        # Build the standard request to feed into the normal inference pipeline
        inference_req = PredictRequest(
            inverter_id=request.inverter_id,
            block="Unknown (Historical)",
            timestamp=request.timestamp,
            features=features
        )
        
        return analyze_inverter_risk(inference_req)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
