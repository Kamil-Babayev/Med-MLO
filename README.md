# Med-MLOps Pipeline

An end-to-end biomedical image classification API with production-ready serving.
Trains a ResNet18 model on PathMNIST (colorectal cancer tissue patches) and serves
predictions via a FastAPI backend with PostgreSQL inference logging.

## Architecture

```
┌─────────────┐     POST /predict      ┌─────────────────┐
│   Client    │ ─────────────────────► │   FastAPI API   │
└─────────────┘                        └────────┬────────┘
                                                │
                                   ┌────────────┴────────────┐
                                   │                         │
                            ┌──────▼──────┐         ┌───────▼──────┐
                            │  ResNet18   │         │  PostgreSQL  │
                            │   Model     │         │  (logging)   │
                            └─────────────┘         └──────────────┘
```

## Quickstart

### Prerequisites

- Docker & Docker Compose
- Python 3.11+

### Run with Docker

```bash
docker compose up --build
```

API will be available at `http://localhost:8000`

### Run locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Postgres
docker compose up db -d

# Train the model
python model/train.py

# Start the API
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/medmlops uvicorn api.main:app --reload
```

## Training

```bash
python model/train.py
```

Model config is in `model/config.yaml`. Trained weights and a full training log
are saved to `model/weights/`.

| Parameter     | Value     |
|---------------|-----------|
| Architecture  | ResNet18  |
| Dataset       | PathMNIST |
| Classes       | 9         |
| Image size    | 64×64     |
| Epochs        | 5         |
| Val accuracy  | ~95.8%    |

## API Endpoints

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | /health    | Liveness check + model version     |
| POST   | /predict   | Upload image, get prediction       |
| GET    | /history   | Paginated inference log            |

Interactive docs available at `http://localhost:8000/docs`

### Example prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@image.png"
```

```json
{
  "filename": "image.png",
  "predicted_class": "colorectal adenocarcinoma epithelium",
  "confidence": 0.9804,
  "model_version": "resnet18-pathmnist-acc0.9577"
}
```

## Testing

```bash
pytest tests/ -v
```

Tests use SQLite in-memory — no Postgres required.

## Project Structure

```
med-mlops/
├── model/
│   ├── train.py          # OOP training script
│   ├── dataset.py        # MedMNIST dataloader
│   ├── config.yaml       # hyperparameters
│   └── weights/          # saved model + training log
├── api/
│   ├── main.py           # FastAPI app
│   ├── schemas.py        # Pydantic models
│   ├── routes/           # endpoint handlers
│   ├── services/         # inference + preprocessing
│   └── db/               # SQLAlchemy models + session
├── tests/                # pytest test suite
├── docker-compose.yml
├── Dockerfile
└── README.md
```
