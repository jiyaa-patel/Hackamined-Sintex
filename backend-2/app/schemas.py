from pydantic import BaseModel
from typing import Dict, List, Any

class FeatureImpact(BaseModel):
    feature: str
    impact: float

class PredictRequest(BaseModel):
    inverter_id: str
    block: str
    timestamp: str | None = None
    features: Dict[str, float]

class PredictHistoricalRequest(BaseModel):
    inverter_id: str
    timestamp: str

class PredictResponse(BaseModel):
    inverter_id: str
    block: str
    risk_score: float
    risk_band: str
    top_factors: List[FeatureImpact]
    narrative_summary: str
    recommended_actions: List[str]

class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] | None = None

class ChatResponse(BaseModel):
    answer: str
    source: str = "Technical Maintenance Manual"
