from fastapi import APIRouter
from api.schemas import HealthResponse
from api.services.inference import ModelService

router = APIRouter()


def create_router(model_service: ModelService) -> APIRouter:
    @router.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            model_loaded=model_service.model is not None,
            model_version=model_service.model_version,
        )

    return router