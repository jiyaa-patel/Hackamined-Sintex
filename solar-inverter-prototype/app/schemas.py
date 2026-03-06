# app/schemas.py
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    inverter_id: str = Field(..., examples=["INV_001"])
    date: str = Field(..., examples=["2026-03-05"])
    features: Dict[str, Optional[float]]

class PredictResponse(BaseModel):
    inverter_id: str
    date: str
    risk_score: float
    prediction: int
    top_factors: list  # [{feature, value, shap}]
    model_version: str = "v1"