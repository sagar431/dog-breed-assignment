from __future__ import annotations

from typing import Sequence

import lightning as L
import timm
import torch
from torch import nn


class DogBreedClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = "mobilenetv3_small_050",
        num_classes: int = 2,
        class_names: Sequence[str] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        if freeze_backbone:
            self._freeze_backbone()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = self._accuracy(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        acc = self._accuracy(logits, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def _freeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        classifier = self.model.get_classifier() if hasattr(self.model, "get_classifier") else None
        if isinstance(classifier, nn.Module):
            for param in classifier.parameters():
                param.requires_grad = True
            return
        for name, param in self.model.named_parameters():
            if any(part in name.lower() for part in ("classifier", "head", "fc")):
                param.requires_grad = True

    @staticmethod
    def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        predictions = torch.argmax(logits, dim=1)
        return (predictions == labels).float().mean()

    @property
    def class_names(self) -> list[str]:
        names = self.hparams.get("class_names")
        if names:
            return list(names)
        return [str(index) for index in range(self.hparams.num_classes)]

