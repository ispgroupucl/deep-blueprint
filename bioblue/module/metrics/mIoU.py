from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.metrics import ConfusionMatrix
import torch


class IoU(ConfusionMatrix):
    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        class_index: Optional[int] = None,
    ):
        super().__init__(
            num_classes,
            normalize=None,
            threshold=threshold,
            compute_on_step=True,
            dist_sync_on_step=False,
        )
        self.class_index = class_index

    def compute(self) -> torch.Tensor:
        cm = super().compute().to(torch.double)
        iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
        if self.class_index is None:
            return iou.mean()
        else:
            return iou[self.class_index]
