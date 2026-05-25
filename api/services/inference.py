import json
import torch
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
from medmnist import INFO

from api.services.preprocess import preprocess_image

DATASET_NAME = "pathmnist"
CLASS_LABELS = list(INFO[DATASET_NAME]["label"].values())


class ModelService:
    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_version = weights_path
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        checkpoint = torch.load(self.weights_path, map_location=self.device)

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_LABELS))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # use val_acc from checkpoint as version string
        val_acc = checkpoint.get("val_acc", 0.0)
        self.model_version = f"resnet18-pathmnist-acc{val_acc:.4f}"

        return model.to(self.device)

    def predict(self, image_bytes: bytes) -> dict:
        tensor = preprocess_image(image_bytes).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted_idx = probs.max(dim=1)

        return {
            "predicted_class": CLASS_LABELS[predicted_idx.item()],
            "confidence": round(confidence.item(), 4),
            "model_version": self.model_version,
        }