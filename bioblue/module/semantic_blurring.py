import pytorch_lightning as pl
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

        return loss

    def validation_step(self, batch, batch_idx):
        seg = batch["segmentation"].to(torch.long)
        img = batch["image"].to(torch.float).unsqueeze(1)

        seg_hat = self(dict(image=img))

        loss = F.cross_entropy(seg_hat, seg)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
