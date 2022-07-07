from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class ResNetClassifierV2(nn.Module):
    def __init__(self, 
                model_cfg,
                input_format,
                output_format,
                classes, 
                architecture=None,
                tune_fc_only=True):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            '18': models.resnet18, '34': models.resnet34,
            '50': models.resnet50, '101': models.resnet101,
            '152': models.resnet152
        }
        

        self.classes = classes
        self.n_classes = len(classes)

        self.architecture = architecture
        
        self.resnet_version=self.architecture['resnet_version']
        self.pretrained = self.architecture['pretrained']
        self.tune_fc_only = self.architecture['tune_fc_only']

        self.fcn_input_format = [ inp for inp in self.input_format if inp != 'image']
        self.input_format = [inp for inp in self.input_format if inp == 'image']

        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[self.resnet_version](pretrained=self.pretrained)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, self.n_classes)

        if tune_fc_only: # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        # print('ResnetClassifierV2', [(key , X[key].shape, X[key].device ) for  key in X.keys()])

        resnet_input = X['image'].repeat(1,3,1,1)

        # print('resnet_input', resnet_input.shape)

        return self.resnet_model(resnet_input)


# class ResNetClassifier(pl.LightningModule):
class ResNetClassifier(nn.Module):
    def __init__(self, 
    
                classes, 
                train_path, 
                vld_path, 
                test_path=None, 
                optimizer='adam', 
                lr=1e-3, 
                resnet_version="50",
                batch_size=16,
                pretrained=True,
                tune_fc_only=True):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        self.classes = classes
        self.n_classes = len(classes)

        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=pretrained)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, self.n_classes)

        if tune_fc_only: # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        return self.resnet_model(X)