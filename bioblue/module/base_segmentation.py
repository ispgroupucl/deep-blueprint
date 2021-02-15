import io
from typing import Any, List
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plF
import torch
import torch.nn.functional as F
import logging

from hydra.utils import instantiate

log = logging.getLogger(__name__)


class BaseSegment(pl.LightningModule):
    def __init__(
        self, segmenter, lr=1e-3, optimizer="torch.optim.Adam", optimizer_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_format = segmenter["input_format"]
        if hasattr(segmenter, "_target_"):
            segmenter = instantiate(segmenter)

        self.segmenter = segmenter
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}

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

        loss = F.cross_entropy(seg_hat, seg)
        iou_val = plF.iou(pl.metrics.utils.to_categorical(seg_hat), seg, num_classes=2)
        return dict(loss=loss, iou=iou_val)

    def training_step(self, batch, batch_idx):
        output = self.common_step(batch)
        self.log(
            "train_loss", output["loss"], on_epoch=True, on_step=False, logger=True
        )

        return output

    def training_epoch_end(self, outputs: List[Any]) -> None:
        ious = torch.stack([tmp["iou"] for tmp in outputs])
        iou_val = torch.mean(ious)
        self.log("train_iou", iou_val, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch)
        self.log("val_loss", output["loss"])
        return output

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        ious = torch.stack([tmp["iou"] for tmp in outputs])
        iou_val = torch.mean(ious)
        self.log("val_iou", iou_val, prog_bar=True)

    def configure_optimizers(self):
        optimizer = {
            "_target_": self.optimizer_class,
            "lr": self.hparams.lr,
            **self.optimizer_params,
        }
        optimizer = instantiate(optimizer, params=self.parameters())
        return optimizer
