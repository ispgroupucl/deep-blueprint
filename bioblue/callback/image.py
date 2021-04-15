import logging
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import cv2
from torchmetrics.functional.classification import iou
from pytorch_lightning.metrics.utils import to_categorical
from bioblue.plot import cm
import numpy as np
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


class PlotImageCallback(pl.Callback):
    def __init__(self, num_samples=32):
        self.num_samples = num_samples
        super().__init__()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        dm = trainer.val_dataloaders[0]
        seen = 0
        for batch in dm:
            segmentation = pl_module(batch)
            if seen > self.num_samples:
                break
            cat_segm = to_categorical(segmentation)
            num_axs = cat_segm.shape[0]
            fig, axs = plt.subplots(ncols=num_axs, figsize=(num_axs * 3, 10))
            for i, ax in enumerate(axs):
                ax.imshow(batch["image"][i].detach(), cmap="gray")
                ax.imshow(
                    cat_segm[i].cpu().detach(),
                    cmap=cm.hsv,
                    alpha=0.7,
                    interpolation="none",
                )
                iou_value = iou(
                    cat_segm[i].cpu().unsqueeze(0),
                    batch["segmentation"][i].unsqueeze(0),
                )
                ax.set_title(f"iou={iou_value*100:.2f}%")
                seen += 1
            plt.show()
            plt.close(fig)


class PlotTrainCallback(pl.Callback):
    def __init__(self, show_percentage=0.1, input="image"):
        self.show = show_percentage
        self.input = input
        self.rng = np.random.default_rng()

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.rng.random() > self.show:
            return
        input_format = pl_module.input_format  # FIXME : create ABC for this
        for input_key in input_format:
            img = batch[input_key]
            bs = img.shape[0]
            fig, axs = plt.subplots(ncols=bs, figsize=(bs * 3, 10))
            for i, ax in enumerate(axs):
                ax.imshow(img[i], cmap="gray")
                ax.set_title(f"{input_key}")
            plt.show()
            plt.close(fig)

        output_format = pl_module.output_format  # FIXME : as above
        for input_key in output_format:
            img = batch[input_key]
            bs = img.shape[0]
            fig, axs = plt.subplots(ncols=bs, figsize=(bs * 3, 10))
            for i, ax in enumerate(axs):
                ax.imshow(batch[self.input][i], cmap="gray")
                ax.imshow(img[i], cmap=cm.hsv, alpha=0.7, interpolation="none")
                ax.set_title(f"{input_key}")
            plt.show()
            plt.close(fig)


class SaveVolumeCallback(pl.Callback):
    def __init__(self, val=True, train=False, test=False):
        self.val = val
        self.train = train
        self.test = test

    def save_dataset(self, pl_module, dataset, type):
        for i, sample in enumerate(tqdm(dataset, desc=f"Saving {type} images")):
            file_index = dataset.reverse_index["image"][i]
            filename = Path(dataset.files[file_index]).stem
            array_index = dataset.array_index["image"][i]
            save_name = Path(f"images/volumes/{filename}/{array_index:04}.jpg")
            unsqueezed_sample = {}
            for k, s in sample.items():
                unsqueezed_sample[k] = torch.tensor(s).unsqueeze(0)
            segm = to_categorical(pl_module(unsqueezed_sample))
            log.debug(f"{segm.device} {segm.shape}")
            segm = segm[0].cpu().to(torch.uint8).numpy()
            save_name.parent.mkdir(parents=True, exist_ok=True)
            original_shape = dataset.original_shape["image"][file_index]
            segm = cv2.resize(segm, original_shape, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(save_name), segm)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.train:
            self.save_dataset(pl_module, trainer.datamodule.train_ds, "train")
        if self.val:
            self.save_dataset(pl_module, trainer.datamodule.val_ds, "validation")
        if self.test:
            self.save_dataset(pl_module, trainer.datamodule.test_ds, "test")
