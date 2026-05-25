import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from tqdm import tqdm

from dataset import get_dataloaders


class Trainer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.train_loader, self.val_loader, num_classes = get_dataloaders(self.config)

        self.model = self._build_model(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5)

        self.best_val_acc = 0.0
        self.history = []

    def _build_model(self, num_classes: int) -> nn.Module:
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
            if self.config["model"]["pretrained"]
            else None
        )
        # Replace the final FC layer to match our number of classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model.to(self.device)

    def _run_epoch(self, loader, training: bool) -> tuple[float, float]:
        self.model.train() if training else self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        label = "Train" if training else "Val"

        with torch.set_grad_enabled(training):
            for images, labels in tqdm(loader, desc=label, leave=False):
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += images.size(0)

        return total_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, val_acc: float):
        os.makedirs(self.config["paths"]["weights_dir"], exist_ok=True)
        path = os.path.join(self.config["paths"]["weights_dir"], "best_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_acc": val_acc,
            "config": self.config,
        }, path)
        print(f"  Checkpoint saved (val_acc={val_acc:.4f})")

    def _save_training_log(self):
        log = {
            "dataset": self.config["dataset"]["name"],
            "architecture": self.config["model"]["architecture"],
            "epochs_run": len(self.history),
            "best_val_acc": self.best_val_acc,
            "trained_at": datetime.utcnow().isoformat(),
            "history": self.history,
        }
        os.makedirs(self.config["paths"]["weights_dir"], exist_ok=True)
        with open(self.config["paths"]["training_log"], "w") as f:
            json.dump(log, f, indent=2)
        print(f"Training log saved to {self.config['paths']['training_log']}")

    def train(self):
        epochs = self.config["training"]["epochs"]
        print(f"\nStarting training for {epochs} epochs...\n")

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")

            train_loss, train_acc = self._run_epoch(self.train_loader, training=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, training=False)
            self.scheduler.step()

            self.history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            })

            print(
                f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f} "
                f"  val_loss={val_loss:.4f}    val_acc={val_acc:.4f}"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)

        self._save_training_log()
        print(f"\nDone. Best val accuracy: {self.best_val_acc:.4f}")


if __name__ == "__main__":
    trainer = Trainer(config_path="model/config.yaml")
    trainer.train()