import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.routes import predict, history, health
from api.services.inference import ModelService
from api.db.models import Base

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "model/weights/best_model.pth")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.db.session import engine
    Base.metadata.create_all(bind=engine)

    model_service = ModelService(weights_path=WEIGHTS_PATH)
    app.include_router(health.create_router(model_service))
    app.include_router(predict.create_router(model_service))
    app.include_router(history.create_router())

    yield


app = FastAPI(
    title="Med-MLOps API",
    description="Biomedical image classifier with inference logging.",
    version="1.0.0",
    lifespan=lifespan,
)