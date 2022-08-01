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
import numpy as np
from tqdm.auto import tqdm
import os
import skimage.io as io
import ipywidgets as widgets
from IPython.display import display
import matplotlib

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

log = logging.getLogger(__name__)


def display_first_channel_batch2(batch, image_key, predictions, gt, mapper):
    image = batch[image_key].cpu()
    bs = image.shape[0]
    
    show_classes = ['0','1']
    # show_classes = ['A','B']
    print(mapper)
    log.debug(image.ndim)
    # print(mapper)
    # print('1')
    if image.ndim == 4:
        # print('2')
        for i in range(bs):
            img = 3500* image[i, 0, :, :].cpu().numpy()
            pred = str(predictions[i].cpu().numpy())
            g = str(gt[i].cpu().numpy()[0])
            # print(mapper[g], mapper[pred])
            if (pred in show_classes) and (g in show_classes) and (g != pred):
                fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(3, 3),  dpi=200)

                normalize = matplotlib.colors.Normalize(vmin=0, vmax=4000)
                ax.imshow(img, cmap='gray', norm = normalize)
                # ax.imshow(img, cmap="gray")
                # ax.set_clim(0,4000)
                
                ax.set_title(f"GT: {mapper[g]}, Pred: {mapper[pred]}")
                plt.show()
                plt.close(fig)


def display_first_channel_batch(batch, image_key, predictions, gt, mapper):
    image = batch[image_key].cpu()
    bs = image.shape[0]
    print(bs)
    # include_index = [i for i in range(bs) if (predictions[i] in list(mapper.keys()) and  gt[i] in list(mapper.keys()))]

    log.debug(image.ndim)
    # print(mapper)
    # print('1')
    if image.ndim == 4:
        # print('2')
        fig, axs = plt.subplots(ncols=bs, figsize=(bs * 5, 10), squeeze=False, dpi=200)
        # fig, axs = plt.subplots(ncols=bs, figsize=(len(include_index) * 5, 10), squeeze=False, dpi=200)
        for i, ax in enumerate(axs[0]):
            img = 3500* image[i, 0, :, :].cpu().numpy()
            pred = str(predictions[i].cpu().numpy())
            g = str(gt[i].cpu().numpy()[0])
            # print(pred,g)
            # print(mapper[pred], mapper[g])
            # print(np.min(img), np.max(img))

            normalize = matplotlib.colors.Normalize(vmin=0, vmax=4000)
            ax.imshow(img, cmap='gray', norm = normalize)
            # ax.imshow(img, cmap="gray")
            # ax.set_clim(0,4000)
            
            ax.set_title(f"GT: {mapper[g]}, Pred: {mapper[pred]}")
        plt.show()
        plt.close(fig)


class ShowClassificationPredictionsCallback(pl.Callback):
    def __init__(self, output_dir) -> None:
        self.output_dir = Path(output_dir)
        
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True, exist_ok=True)


    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.sanity_checking:
            return
       
        used_classes = trainer.test_dataloaders[0].dataset.classes_mapper
        used_classes_lst = used_classes.items()
        inverted_mapper = {str(v): k  for k,v in used_classes_lst }
        
        # print(used_classes)

        classification = batch['class']

        classification_hat = pl_module(batch)
        # print(classification, 'vs', classification_hat)

        classification_hat = F.softmax(classification_hat, dim=1)
        # print(classification, 'vs', classification_hat)
        classification_hat = torch.argmax(classification_hat,dim=1)
        # print(classification, 'vs', classification_hat)

        
        # images_batch = batch['image']
        # print(images_batch.shape)


        display_first_channel_batch2(batch, 'image', classification_hat, classification, inverted_mapper)
        # display_first_channel_batch(batch, 'image', classification_hat, classification, inverted_mapper)


class ClassificationConfusionMatrixCallback(pl.Callback):
    def __init__(self, normalize=None) -> None:
        assert (normalize == None ) or (normalize =='true') or (normalize == 'pred') or (normalize == 'all')

        self.normalize = normalize


    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        
        self.total_predictions = []
        self.total_gt = []


    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if trainer.sanity_checking:
            return
       
        used_classes = trainer.test_dataloaders[0].dataset.classes_mapper
        used_classes_lst = used_classes.items()
        inverted_mapper = {str(v): k  for k,v in used_classes_lst }
        
        # print(used_classes)

        classification = batch['class']

        classification_hat = pl_module(batch)
        # print(classification, 'vs', classification_hat)

        classification_hat = F.softmax(classification_hat, dim=1)
        # print(classification, 'vs', classification_hat)
        classification_hat = torch.argmax(classification_hat,dim=1)
        # print(classification, 'vs', classification_hat)

        self.total_gt.append(classification.cpu().numpy())
        self.total_predictions.append(np.expand_dims(classification_hat.cpu().numpy(),axis=1))

        # print('gt: ', self.total_gt[-1].shape)
        # print('pred: ', self.total_predictions[-1].shape)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        used_classes = trainer.test_dataloaders[0].dataset.classes_mapper
        class_names  = list(used_classes.keys())

        y_true = np.concatenate(self.total_gt, axis=0).ravel()
        y_pred = np.concatenate(self.total_predictions, axis=0).ravel()

        # print('test_end')
        # print(y_true.shape)
        # print(y_pred.shape)

        conf_mat = confusion_matrix(y_true,y_pred, normalize=self.normalize )

        # print(conf_mat)

        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=200)

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                                        display_labels=class_names,
                                        )
                                    
        disp.plot(ax=ax)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        fig.show()
        



    