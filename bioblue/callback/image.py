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
    def __init__(self, show_percentage=0.1, input="image", val=False):
        self.show = show_percentage
        self.input = input
        self.rng = np.random.default_rng()
        self.shown = False
        self.val = val

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.val:
            self.on_train_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not self.shown:
            return
        log.debug(outputs)
        img = batch[self.input]
        bs = img.shape[0]
        segmentation = pl_module(batch)
        segmentation = to_categorical(segmentation).cpu()
        fig, axs = plt.subplots(ncols=bs, figsize=(bs * 5, 10))
        for i, ax in enumerate(axs):
            ax.imshow(img[i], cmap="gray")
            ax.imshow(segmentation[i], cmap=cm.hsv, alpha=0.7, interpolation="none")
            ax.set_title(f"output")
        plt.show()
        plt.close(fig)

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.val:
            self.on_train_batch_start(
                trainer, pl_module, batch, batch_idx, dataloader_idx
            )

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.rng.random() > self.show:
            self.shown = False
            return
        self.shown = True
        input_format = pl_module.input_format  # FIXME : create ABC for this
        for input_key in input_format:
            img = batch[input_key]
            bs = img.shape[0]
            fig, axs = plt.subplots(ncols=bs, figsize=(bs * 5, 10))
            for i, ax in enumerate(axs):
                ax.imshow(img[i], cmap="gray")
                ax.set_title(f"{input_key}")
            plt.show()
            plt.close(fig)

        output_format = pl_module.output_format  # FIXME : as above
        for input_key in output_format:
            img = batch[input_key]
            bs = img.shape[0]
            log.debug(f"segmentation : {np.unique(img)}")
            fig, axs = plt.subplots(ncols=bs, figsize=(bs * 5, 10))
            for i, ax in enumerate(axs):
                ax.imshow(batch[self.input][i], cmap="gray")
                ax.imshow(img[i], cmap=cm.hsv, alpha=0.7, interpolation="none")
                title = batch["_title"][i] if "_title" in batch else ""
                ax.set_title(f"{input_key} {title}")
            plt.show()
            plt.close(fig)


class SaveVolumeCallback(pl.Callback):
    def __init__(self, val=True, train=True, test=False):
        self.val = val
        self.train = train
        self.test = test
        self.period = 5
        self.ds_files = None
        self.ds_reverse_index = None
        self.ds_array_index = None

    def on_fit_start(self, trainer, pl_module):
        dataset = trainer.datamodule.val_ds
        dataset.initialize()
        self.ds_files = dataset.files
        self.ds_array_index = dataset.array_index
        self.ds_reverse_index = dataset.reverse_index
        if hasattr(dataset, "reset"):
            dataset.reset()

    def save_dataset(self, pl_module, dataset, type):
        for i, sample in enumerate(tqdm(dataset, desc=f"Saving {type} images")):
            file_index = dataset.reverse_index["image"][i]
            filename = Path(dataset.files[file_index]).stem
            array_index = dataset.array_index["image"][i]
            save_name = Path(
                f"images/volumes/{type}/{filename}"
            )  # / {array_index:04}.png")
            unsqueezed_sample = {}
            for k, s in sample.items():
                unsqueezed_sample[k] = torch.tensor(s).unsqueeze(0)
            segm = to_categorical(pl_module(unsqueezed_sample))
            log.debug(f"{segm.device} {segm.shape}")
            segm = segm[0].cpu().to(torch.uint8).numpy()
            save_name.parent.mkdir(parents=True, exist_ok=True)
            if segm.ndim == 2:
                write_image(save_name / f"{array_index:04}.png", segm)
            else:
                write_volume(save_name, segm)
            # original_shape = dataset.original_shape["image"][file_index]
            # segm = cv2.resize(segm, original_shape, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(save_name), segm)

        if hasattr(dataset, "reset"):
            dataset.reset()

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.running_sanity_check:
            return
        epoch = trainer.current_epoch
        log.debug(f"train epoch {epoch}")
        if epoch % self.period == 0:
            batch_size = batch["segmentation"].shape[0]
            start_idx = batch_idx * batch_size
            segmentation = pl_module(batch)
            segmentation = to_categorical(segmentation).cpu().to(torch.uint8).numpy()
            log.debug(f"{batch_idx} {start_idx}")
            log.debug(f"{segmentation.shape} {np.max(segmentation)}")
            for i, segm in enumerate(segmentation, start=start_idx):
                log.debug(f"inner shape {segm.shape} {np.unique(segm)}")
                file_index = self.ds_reverse_index["image"][i]
                filename = Path(self.ds_files[file_index]).stem
                array_index = self.ds_array_index["image"][i]
                save_name = Path(
                    f"images/volumes/epoch{epoch}/{filename}"  # /{array_index:04}.png"
                )
                if segm.ndim == 2:
                    write_image(save_name / f"{array_index:04}.png", segm)
                elif segm.ndim == 3:
                    write_volume(save_name, segm)

                log.debug(f"written image or volume to {save_name}")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.train:
            self.save_dataset(pl_module, trainer.datamodule.train_ds, "train")
        if self.val:
            self.save_dataset(pl_module, trainer.datamodule.val_ds, "validation")
        if self.test:
            self.save_dataset(pl_module, trainer.datamodule.test_ds, "test")


def write_image(filename, image):
    filename.parent.mkdir(parents=True, exist_ok=True)
    res = cv2.imwrite(str(filename), image)
    if res is False:
        raise RuntimeError("CV2 imwrite failed to save image.")


def write_volume(dirpath, volume):
    for index in range(volume.shape[-1]):
        filename = dirpath / f"{index:04}.png"
        write_image(filename, volume[:, :, index])

