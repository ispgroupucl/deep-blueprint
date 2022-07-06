import logging
from typing import Any, Optional
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
from torchmetrics.functional.classification import iou
from torchmetrics.utilities.data import to_categorical
from bioblue.plot import cm
import numpy as np
from tqdm.auto import tqdm
import os
import skimage.io as io
from itkwidgets import view as view3d
import ipywidgets as widgets
from IPython.display import display

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
            segmentation = pl_module.predict(batch)
            if seen > self.num_samples:
                break
            cat_segm = to_categorical(segmentation)
            bs = cat_segm.shape[0]
            batch["_segm"] = cat_segm
            display_batch(batch, "image", "_segm")
            seen += bs


class InputHistoCallback(pl.Callback):
    def __init__(self, show_percentage=0.1, inputs=("image",), val=False) -> None:
        self.show = show_percentage
        self.inputs = inputs
        self.rng = np.random.default_rng()
        self.shown = False
        self.val = val

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.on_train_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.rng.random() > self.show:
            return
        fig, axs = plt.subplots(len(self.inputs), figsize=(10, 3), squeeze=False)
        for ax, input in zip(axs[0], self.inputs):
            flat_batch = batch[input].cpu().flatten().to(torch.uint8).numpy()
            ax.hist(flat_batch, bins=256, range=(0, 255))
        plt.show()
        plt.close(fig)


class PlotTrainCallback(pl.Callback):
    def __init__(self, show_percentage=0.1, input="image", val=False):
        self.show = show_percentage
        self.input = input
        self.rng = np.random.default_rng()
        self.shown = False
        self.val = val


    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) :
        if self.val:
            self.on_train_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )
   
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
        img = batch[self.input].cpu()
        bs = img.shape[0]
        
        for dtype in batch:
            batch[dtype] = batch[dtype].squeeze(1).to(torch.float)
        segmentation = pl_module.predict(batch) 
        # segmentation = pl_module.predict(torch.squeeze(batch,1))       
        # print("on_train_batch_end: ", segmentation.shape, torch.unique(segmentation[0]))
        # segmentation = F.softmax(segmentation, dim=1)
        # print("on_train_batch_end AFTER SOFTMAX: ", segmentation.shape, torch.unique(segmentation[0]))

        segmentation = to_categorical(segmentation,argmax_dim=1).cpu()
        # print("on_train_batch_end AFTER to_categorical: ", segmentation.shape, torch.unique(segmentation[0]))

        # segmentation = to_categorical(segmentation).cpu()
        batch["_segm"] = segmentation
        display_batch(batch, self.input, "_segm")
        del batch["_segm"]  # really needed ?

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.val:
            self.on_train_batch_start(
                trainer, pl_module, batch, batch_idx, dataloader_idx
            )

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
            display_batch(batch, image_key=input_key)

        output_format = pl_module.output_format  # FIXME : as above
        for output_key in output_format:
            display_batch(batch, image_key=self.input, segm_key=output_key)


def display_batch(batch, image_key, segm_key=None, title_key="_title"):
    image = batch[image_key].cpu()
    bs = image.shape[0]
    log.debug(image.ndim)
    if image.ndim == 3:
        s = ...
    elif image.ndim == 4:
        s = image.shape[1] // 2
    if image.ndim == 3:
        fig, axs = plt.subplots(ncols=bs, figsize=(bs * 5, 10), squeeze=False)
        for i, ax in enumerate(axs[0]):
            img = image[i, :, s, :].cpu()
            ax.imshow(img, cmap="gray")
            key_title = image_key
            if segm_key is not None:
                mask = batch[segm_key][i, :, s, :].cpu()
                ax.imshow(mask, cmap=cm.hsv, alpha=0.7, interpolation="none")
                key_title = segm_key
            title = batch[title_key][i] if title_key in batch else ""
            ax.set_title(f"{key_title} {title}")
        plt.show()
        plt.close(fig)
    elif image.ndim == 4:
        children = []
        for i in range(bs):
            img = image[i].cpu()
            if segm_key is not None:
                mask = batch[segm_key][i].cpu()
            else:
                mask = None
            children.append(view3d(image=img, label_image=mask))
        tabs = widgets.Tab(children)
        display(tabs)

       
class SaveVolumeCallback(pl.Callback):
    def __init__(self, val=True, train=True, test=False, period=5):
        self.val = val
        self.train = train
        self.test = test
        self.period = period
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
            segm = to_categorical(pl_module.predict(unsqueezed_sample))
            log.debug(f"{segm.device} {segm.shape}")
            segm = segm[0].cpu().to(torch.uint8).numpy()
            save_name.parent.mkdir(parents=True, exist_ok=True)
            if segm.ndim == 2:
                write_image(save_name / f"{array_index:04}.png", segm)
            else:
                write_volume(save_name, segm)
            # original_shape = dataset.original_shape["image"][file_index]
            # segm = cv2.resize(segm, original_shape, interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite(str(save_name), segm)

        if hasattr(dataset, "reset"):
            dataset.reset()

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        log.debug(f"train epoch {epoch}")
        if epoch % self.period == 0:
            batch_size = batch["segmentation"].shape[0]
            start_idx = batch_idx * batch_size
            segmentation = pl_module.predict(batch)
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


class SavePredictionMaskCallback(pl.Callback):
    def __init__(self, output_dir, max_batch_size) :
        self.output_dir = Path(output_dir)
        self.max_batch_size = max_batch_size
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
        # if trainer.running_sanity_check:
        #     return    
        if trainer.sanity_checking:
            return
        
        batch["segmentation"] = batch["segmentation"][0]
        
        segmentation = pl_module(batch)

        segmentation = to_categorical(segmentation).cpu().to(torch.uint8).numpy()

        dataset = trainer.datamodule.test_ds
        
        cur_batch_size = batch["segmentation"].shape[0]

        for i in range(cur_batch_size):
            idx = batch_idx * self.max_batch_size + i
            name = (dataset.files[idx]["name"]).split(".")[0]+ ".png"

            io.imsave(self.output_dir / name, segmentation[i], check_contrast=False)


class SavePredictionMaskCallback2(pl.Callback):
    def __init__(self, output_dir, max_batch_size) :
        self.output_dir = Path(output_dir)
        # print(self.output_dir.stem)
        self.max_batch_size = max_batch_size
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reconstructed_image = []

    def on_predict_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.sanity_checking:
            return
        

        batch["segmentation"] = batch["segmentation"]
        for dtype in batch:
            # print(batch[dtype].shape)
            batch[dtype]=torch.squeeze(batch[dtype],1)

        segmentation = pl_module(batch)

        segmentation = to_categorical(segmentation).cpu().to(torch.uint8).numpy()

        dataset = trainer.datamodule.test_ds
        grid_size = dataset.grid_size
        
        cur_batch_size = batch["segmentation"].shape[0]


        for i in range(cur_batch_size):
            idx = batch_idx * self.max_batch_size + i

            self.reconstructed_image.append(segmentation[i])
            # print(len(self.reconstructed_image))
            if len(self.reconstructed_image) == grid_size:
                index = idx // grid_size
                name = (dataset.files[index]["name"]).split(".")[0]+ ".png"

                side = int(self.reconstructed_image[0].shape[0] * np.sqrt(grid_size))
                # print(side)
                dump = np.zeros((side,side)).astype(np.uint8)
                for j, patch in enumerate(self.reconstructed_image):
                    a = j // int(np.sqrt(grid_size))
                    b = j % int(np.sqrt(grid_size))
                    dump[
                            a*patch.shape[0]: (a+1)*patch.shape[0],
                            b*patch.shape[1]: (b+1)*patch.shape[1]
                        ] = patch
                
                # print(np.unique(dump))
                # print(self.output_dir)
                io.imsave(self.output_dir / name, dump, check_contrast=False)
                self.reconstructed_image = []
            
            # name = (dataset.files[idx]["name"]).split(".")[0]+ ".png"
            # io.imsave(self.output_dir / name, segmentation[i], check_contrast=False)
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
        if trainer.sanity_checking:
            return
        

        batch["segmentation"] = batch["segmentation"][0]
        
        segmentation = pl_module(batch)

        segmentation = to_categorical(segmentation).cpu().to(torch.uint8).numpy()

        dataset = trainer.datamodule.test_ds
        grid_size = dataset.grid_size
        
        cur_batch_size = batch["segmentation"].shape[0]


        for i in range(cur_batch_size):
            idx = batch_idx * self.max_batch_size + i

            self.reconstructed_image.append(segmentation[i])
            # print(len(self.reconstructed_image))
            if len(self.reconstructed_image) == grid_size:
                index = idx // grid_size
                name = (dataset.files[index]["name"]).split(".")[0]+ ".png"

                side = int(self.reconstructed_image[0].shape[0] * np.sqrt(grid_size))
                # print(side)
                dump = np.zeros((side,side)).astype(np.uint8)
                for j, patch in enumerate(self.reconstructed_image):
                    a = j // int(np.sqrt(grid_size))
                    b = j % int(np.sqrt(grid_size))
                    dump[
                            a*patch.shape[0]: (a+1)*patch.shape[0],
                            b*patch.shape[1]: (b+1)*patch.shape[1]
                        ] = patch
                
                # print(np.unique(dump))
                # print(self.output_dir)
                io.imsave(self.output_dir / name, dump, check_contrast=False)
                self.reconstructed_image = []
            
            # name = (dataset.files[idx]["name"]).split(".")[0]+ ".png"
            # io.imsave(self.output_dir / name, segmentation[i], check_contrast=False)
            
def write_image(filename, image):
    filename.parent.mkdir(parents=True, exist_ok=True)
    res = cv2.imwrite(str(filename), image)
    if res is False:
        raise RuntimeError("CV2 imwrite failed to save image.")


def write_volume(dirpath, volume):
    for index in range(volume.shape[-1]):
        filename = dirpath / f"{index:04}.png"
        write_image(filename, volume[:, :, index])

