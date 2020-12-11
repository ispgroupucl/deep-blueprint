import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plF
from pytorch_lightning.metrics.utils import to_categorical, to_onehot
import torch
import torch.nn.functional as F

from hydra.utils import instantiate


class SemanticBlur(pl.LightningModule):
    def __init__(self, segmenter, kernel_size=3, lr=1e-3, ssl=False):
        super().__init__()
        self.save_hyperparameters()
        if hasattr(segmenter, "_target_"):
            segmenter = instantiate(segmenter)

        self.segmenter = segmenter
        self.kernel_size = kernel_size
        self.ssl = ssl

    def forward(self, x):
        seg = self.segmenter(x)[0]

        return seg

    def training_step(self, batch, batch_idx):
        seg = batch["segmentation"].to(torch.long)  # .to(torch.float)
        img = batch["image"].to(torch.float)
        img = img.unsqueeze(1)
        seg_hat = self(dict(image=img))
        if self.ssl:
            # TODO: fix for multiple classes ?
            seg_hat = F.softmax(seg_hat)
            kernel = torch.ones(
                (img.shape[0], seg_hat.shape[1], self.kernel_size, self.kernel_size),
                device=self.device,
            )
            smooth_img = torch.sum(
                F.conv2d(img * seg_hat, kernel, padding=self.kernel_size // 2),
                axis=1,
                keepdim=True,
            )
            loss = F.mse_loss(smooth_img, img)
        else:
            loss = F.cross_entropy(seg_hat, seg)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        iou_val = plF.iou(to_categorical(seg_hat), seg, num_classes=2)
        return dict(loss=loss, iou=iou_val)

    def training_epoch_end(self, outputs) -> None:
        ious = torch.stack([tmp["iou"] for tmp in outputs])
        iou_val = torch.mean(ious)
        self.log("train_iou", iou_val, prog_bar=True, logger=True)

    # def on_train_end(self) -> None:
    #     if self.trainer.checkpoint_callback is None:
    #         return

    #     if len(self.trainer.checkpoint_callback.best_model_path) > 0:
    #         self.logger.experiment.log_artifacts(
    #             run_id=self.logger.run_id, local_dir="./models", artifact_path="models"
    #         )

    def validation_step(self, batch, batch_idx):
        seg = batch["segmentation"].to(torch.long)
        img = batch["image"].to(torch.float).unsqueeze(1)

        seg_hat = self(dict(image=img))

        loss = F.cross_entropy(seg_hat, seg)
        self.log(
            "val_loss", loss, prog_bar=False, on_step=False, on_epoch=True, logger=True
        )
        return dict(loss=loss, pred=seg_hat, target=seg)

    def validation_epoch_end(self, outputs) -> None:
        preds = to_categorical(torch.cat([tmp["pred"] for tmp in outputs]))
        targets = torch.cat([tmp["target"] for tmp in outputs])

        iou_val = plF.iou(preds, targets, num_classes=2)
        self.log("val_iou", iou_val, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
