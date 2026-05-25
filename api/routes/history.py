from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.schemas import HistoryResponse, InferenceRecord
from api.db.session import get_db
from api.db.models import Inference

router = APIRouter()


def create_router() -> APIRouter:
    @router.get("/history", response_model=HistoryResponse)
    def history(
        page: int = Query(1, ge=1),
        page_size: int = Query(10, ge=1, le=100),
        db: Session = Depends(get_db),
    ):
        total = db.query(Inference).count()
        results = (
            db.query(Inference)
            .order_by(Inference.timestamp.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        return HistoryResponse(
            total=total,
            page=page,
            page_size=page_size,
            results=[InferenceRecord.model_validate(r) for r in results],
        )

    return router