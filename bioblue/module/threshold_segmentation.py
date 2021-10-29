import io
from typing import Any, List
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plF
import torch
import torch.nn.functional as F
import logging
from torchmetrics.utilities.data import to_categorical

from hydra.utils import instantiate

log = logging.getLogger(__name__)


class ThresholdSegment(pl.LightningModule):
    def __init__(
        self,
        segmenter,
        kernel_size=3,
        lr=1e-3,
        ssl=False,
        optimizer="torch.optim.Adam",
        optimizer_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if hasattr(segmenter, "_target_"):
            segmenter = instantiate(segmenter)

        self.segmenter = segmenter
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.kernel_size = kernel_size
        self.ssl = ssl

    def forward(self, x):
        x["image"] = x["image"].unsqueeze(1)
        seg = self.segmenter(x)[0]

        return seg

    def common_step(self, batch):
        seg = batch["segmentation"].to(torch.long)  # .to(torch.float)
        img = batch["image"].to(torch.float)
        seg_hat = self(dict(image=img))

        loss = F.cross_entropy(seg_hat, seg)
        iou_val = plF.iou(to_categorical(seg_hat), seg, num_classes=2)
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
