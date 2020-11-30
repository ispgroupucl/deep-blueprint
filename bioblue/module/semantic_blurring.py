import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plF
import torch
import torch.nn.functional as F

from hydra.utils import instantiate


class SemanticBlur(pl.LightningModule):
    def __init__(self, segmenter, kernel_size=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        if hasattr(segmenter, "_target_"):
            segmenter = instantiate(segmenter)

        self.segmenter = segmenter
        self.kernel_size = kernel_size

    def forward(self, x):
        seg = self.segmenter(x)[0]

        return seg

    def training_step(self, batch, batch_idx):
        seg = batch["segmentation"].to(torch.long)  # .to(torch.float)
        img = batch["image"].to(torch.float)
        img = img.unsqueeze(1)
        seg_hat = self(dict(image=img))
        # kernel = torch.ones(self.kernel_size, device=self.device)
        # smooth_img = torch.sum(F.conv2d(img * seg, kernel), axis=1)
        loss = F.cross_entropy(seg_hat, seg)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True
        )
        return loss

    def on_train_end(self) -> None:
        if self.trainer.checkpoint_callback is None:
            return

        if len(self.trainer.checkpoint_callback.best_model_path) > 0:
            print("sending checkpoing to logger")
            self.logger.experiment.log_artifact(
                run_id=self.logger.run_id,
                local_path=self.trainer.checkpoint_callback.best_model_path,
            )

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
        preds = plF.to_categorical(torch.cat([tmp["pred"] for tmp in outputs]))
        targets = torch.cat([tmp["target"] for tmp in outputs])

        iou_val = plF.iou(preds, targets, num_classes=2)
        self.log("iou", iou_val, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
