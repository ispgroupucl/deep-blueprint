import io
from typing import Any, List
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plF
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import logging

from .metrics import IoU

from hydra.utils import instantiate

log = logging.getLogger(__name__)


class BaseSegment(pl.LightningModule):
    def __init__(
        self,
        segmenter,
        lr=1e-3,
        optimizer="torch.optim.Adam",
        optimizer_params=None,
        scheduler=None,
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_format = segmenter["input_format"]
        if hasattr(segmenter, "_target_"):
            segmenter = instantiate(segmenter)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.segmenter = segmenter
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.scheduler = scheduler
        self.iou = nn.ModuleDict()
        for set in ["train", "val"]:
            self.iou[f"{set}_mean"] = IoU(num_classes=2)
            self.iou[f"{set}_bg"] = IoU(num_classes=2, class_index=0)
            self.iou[f"{set}_fg"] = IoU(num_classes=2, class_index=1)

    def forward(self, x):
        for dtype in x:
            x[dtype] = x[dtype].unsqueeze(1)
        seg = self.segmenter(x)[0]

        return seg

    def common_step(self, batch):
        seg = batch["segmentation"].to(torch.long)  # .to(torch.float)
        seg[seg == 2] = 0
        input_sample = {}
        for dtype in self.input_format:
            input_sample[dtype] = batch[dtype].to(torch.float)
        seg_hat = self(input_sample)

        return seg, seg_hat

    def training_step(self, batch, batch_idx):
        seg, seg_hat = self.common_step(batch)
        loss = self.loss(seg_hat, seg)
        self.log("train_loss", loss, on_epoch=True, on_step=False, logger=True)
        for name, iou in self.iou.items():
            if "train" not in name:
                continue
            iou(seg_hat, seg)
            progbar = name == "train_mean"
            self.log(f"{name}iou", iou, prog_bar=progbar)
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        seg, seg_hat = self.common_step(batch)
        loss = self.loss(seg_hat, seg)
        self.log("val_loss", loss)
        for name, iou in self.iou.items():
            if "val" not in name:
                continue
            iou(seg_hat, seg)
            progbar = name == "val_mean"
            self.log(f"{name}iou", iou, prog_bar=progbar)
        return dict(loss=loss)

    def configure_optimizers(self):
        optimizer = {
            "_target_": self.optimizer_class,
            "lr": self.hparams.lr,
            **self.optimizer_params,
        }
        optimizer = instantiate(optimizer, params=self.parameters())
        if self.scheduler is not None:
            scheduler = instantiate(self.scheduler, optimizer=optimizer)
        else:
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                threshold=1e-3,
                threshold_mode="abs",
            )
        return dict(
            optimizer=optimizer,
            lr_scheduler={
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
                "monitor": "val_loss",
            },
        )
