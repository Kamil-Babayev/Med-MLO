from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from api.schemas import PredictionResponse
from api.services.inference import ModelService
from api.db.session import get_db
from api.db.models import Inference

router = APIRouter()


def create_router(model_service: ModelService) -> APIRouter:
    @router.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
    ):
        if file.content_type not in ("image/jpeg", "image/png"):
            raise HTTPException(status_code=422, detail="Only JPEG and PNG images are supported.")

        image_bytes = await file.read()
        result = model_service.predict(image_bytes)

        # log to DB
        record = Inference(
            timestamp=datetime.now(timezone.utc),
            filename=file.filename,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            model_version=result["model_version"],
        )
        db.add(record)
        db.commit()

        return PredictionResponse(
            filename=file.filename,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            model_version=result["model_version"],
        )

    return router