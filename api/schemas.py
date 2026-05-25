from datetime import datetime
from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    model_version: str


class InferenceRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    filename: str
    predicted_class: str
    confidence: float
    model_version: str


class HistoryResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[InferenceRecord]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str