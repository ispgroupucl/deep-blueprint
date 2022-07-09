import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim import lr_scheduler

import logging
import numpy as np

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from hydra.utils import instantiate

class BaseClassifier(pl.LightningModule):
    def __init__(
                self,
                classifier,
                lr=1e-3, 
                optimizer='torch.optim.Adam', 
                loss=nn.CrossEntropyLoss,
                optimizer_params=None,
                scheduler=None,
                class_weights = None,
                batch_size=16,
                transfer=True, 
                tune_fc_only=True
                ):
        super().__init__()

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.input_format = classifier["input_format"]
        self.output_format = classifier["output_format"]

        # print(classifier)

        if hasattr(classifier, "_target_"):
            classifier = instantiate(classifier)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)

        # print(classifier)

        self.classifier = classifier
        self.classes = self.classifier.classes

        # print(self.classes)

        self.num_classes = len(self.classes)
        
        # print(dict(self.classifier.named_children()).keys())
        # print(dict(self.classifier.named_children())['input_process'])

        if loss['_target_'] == "torch.nn.CrossEntropyLoss":
            self.loss =instantiate(loss,weight=class_weights)
        else:
            self.loss = instantiate(loss)

        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.scheduler = scheduler

        # #instantiate loss criterion
        # self.criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        # # Using a pretrained ResNet backbone
        # self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        # linear_size = list(self.resnet_model.children())[-1].in_features
        # # replace final layer for fine tuning
        # self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        # if tune_fc_only: # option to only tune the fully-connected layers
        #     for child in list(self.resnet_model.children())[:-1]:
        #         for param in child.parameters():
        #             param.requires_grad = False

    def predict(self, X):
        X = self.transfer_batch_to_device(X, self.device, 0)
        return self(X)

    def forward(self, X):
        # print('classifier_forward')
        # return self.classifier(X)
        for dtype in X:
            if len(X[dtype].shape) == 3:
                X[dtype] = X[dtype].unsqueeze(1).to(torch.float)
            if len(X[dtype].shape) == 4:
                X[dtype] = X[dtype].to(torch.float)
        classif = self.classifier(X)
        return classif


    def common_step(self, batch):
        # print('common_step')
        # classif = batch["class"]
        classif = torch.squeeze(batch["class"],1)

        input_sample = {}
        for dtype in self.input_format:
            input_sample[dtype] = batch[dtype].to(torch.float)

        # print('common_step: compute_hat')
        classif_hat = self(input_sample)

        # print(f'common_step results: {classif.shape} (GT), {classif_hat.shape} (pred)')

        return classif, classif_hat
    
    def training_step(self, batch, batch_idx):
        classif, classif_hat = self.common_step(batch)

        if self.num_classes == 2:
            classif = F.one_hot(classif, num_classes=2).float()
        
        loss = self.loss(classif_hat, classif)

        # print("loss", loss)
        # print("classif: GT", classif, ' VS ', classif_hat )
        # print(" unsqu classif GT", torch.unsqueeze(classif,1), ' VS ', classif_hat ) 
        # print("argmax classif GT", torch.argmax(torch.unsqueeze(classif,1),1), ' VS ', torch.argmax(classif_hat,1) ) 
        # print("max classif GT", torch.max(torch.unsqueeze(classif,1),1), ' VS ', torch.max(classif_hat,1) ) 
        # print()

        acc = (torch.argmax(torch.unsqueeze(classif,1),1) == torch.argmax(classif_hat,1)) \
                .type(torch.FloatTensor).mean()
        # acc = (torch.argmax(classif,1) == torch.argmax(classif_hat,1)) \
        #         .type(torch.FloatTensor).mean()

        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dict(loss=loss)


        # x, y = batch
        # preds = self(x)
        # if self.num_classes == 2:
        #     y = F.one_hot(y, num_classes=2).float()
        
        # loss = self.criterion(preds, y)
        # acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
        #         .type(torch.FloatTensor).mean()
        # # perform logging
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # return loss

    
    def validation_step(self, batch, batch_idx):
        classif, classif_hat = self.common_step(batch)

        if self.num_classes == 2:
            classif = F.one_hot(classif, num_classes=2).float()
        
        # print(classif.shape, classif_hat.shape)

        loss = self.loss(classif_hat, classif)
        acc = (torch.argmax(torch.unsqueeze(classif,1),1) == torch.argmax(classif_hat,1)) \
                .type(torch.FloatTensor).mean()
        # acc = (torch.argmax(classif,1) == torch.argmax(classif_hat,1)) \
        #         .type(torch.FloatTensor).mean()

        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

        # x, y = batch
        # preds = self(x)
        # if self.num_classes == 2:
        #     y = F.one_hot(y, num_classes=2).float()
        
        # loss = self.criterion(preds, y)
        # acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
        #         .type(torch.FloatTensor).mean()
        # # perform logging
        # self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    
    def test_step(self, batch, batch_idx):
        classif, classif_hat = self.common_step(batch)

        if self.num_classes == 2:
            classif = F.one_hot(classif, num_classes=2).float()
        
        loss = self.loss(classif_hat, classif)
        acc = (torch.argmax(torch.unsqueeze(classif,1),1) == torch.argmax(classif_hat,1)) \
                .type(torch.FloatTensor).mean()
        # acc = (torch.argmax(classif,1) == torch.argmax(classif_hat,1)) \
        #         .type(torch.FloatTensor).mean()
        
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
                
        # x, y = batch
        # preds = self(x)
        # if self.num_classes == 2:
        #     y = F.one_hot(y, num_classes=2).float()
        
        # loss = self.criterion(preds, y)
        # acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
        #         .type(torch.FloatTensor).mean()
        # # perform logging
        # self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        # self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)


    def configure_optimizers(self):        
        optimizer = {
            "_target_": self.optimizer_class,
            "lr": self.hparams.lr,
            **self.optimizer_params,
        }
        optimizer = instantiate(optimizer, params=self.parameters())
        if self.scheduler is not None:
            scheduler = instantiate(self.scheduler, optimizer=optimizer)
        else:
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                threshold=1e-3,
                threshold_mode="abs",
            )
        return dict(
            optimizer=optimizer,
            lr_scheduler={
                "scheduler": scheduler,
                "interval": "step",
                # "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
                "monitor": "val_loss",
            },
        )
        
        
        # return self.optimizer(self.parameters(), lr=self.lr)

    # def train_dataloader(self):
    #     # values here are specific to pneumonia dataset and should be changed for custom data
    #     transform = transforms.Compose([
    #             transforms.Resize((500,500)),
    #             transforms.RandomHorizontalFlip(0.3),
    #             transforms.RandomVerticalFlip(0.3),
    #             transforms.RandomApply([
    #                 transforms.RandomRotation(180)                    
    #             ]),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.48232,), (0.23051,))
    #     ])
    #     img_train = ImageFolder(self.train_path, transform=transform)
    #     return DataLoader(img_train, batch_size=self.batch_size, shuffle=True)

    # def val_dataloader(self):
    #     # values here are specific to pneumonia dataset and should be changed for custom data
    #     transform = transforms.Compose([
    #             transforms.Resize((500,500)),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.48232,), (0.23051,))
    #     ])
        
    #     img_val = ImageFolder(self.vld_path, transform=transform)
        
    #     return DataLoader(img_val, batch_size=1, shuffle=False)


    # def test_dataloader(self):
    #     # values here are specific to pneumonia dataset and should be changed for custom data
    #     transform = transforms.Compose([
    #             transforms.Resize((500,500)),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.48232,), (0.23051,))
    #     ])
        
    #     img_test = ImageFolder(self.test_path, transform=transform)
        
    #     return DataLoader(img_test, batch_size=1, shuffle=False)
