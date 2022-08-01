from re import I
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MobileNetClassifier(nn.Module):
    def __init__(self, 
                # model_cfg,
                input_format,
                output_format,
                classes, 
                architecture=None,
                ):
        super().__init__()

        self.__dict__.update(locals())
        mobilenets = {
            'v1': models.mobilenet, 'v2': models.mobilenet_v2,
            'v3_small': models.mobilenet_v3_small,
            'v3_large': models.mobilenet_v3_large,
        }
        

        self.classes = classes
        self.n_classes = len(classes)

        self.architecture = architecture
        
        self.mobilenet_version=self.architecture['mobilenet_version']
        self.pretrained = self.architecture['pretrained']

        self.input_format = input_format

        self.fcn_input_format = [ inp for inp in self.input_format if inp != 'image']
        self.input_format = [inp for inp in self.input_format if inp == 'image']

        if self.pretrained:
            self.model = mobilenets[self.mobilenet_version](pretrained = self.pretrained)
            in_feats = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features=in_feats, out_features=self.n_classes)

        else:
            # Using a pretrained MobileNet backbone
            self.model = mobilenets[self.mobilenet_version](num_classes=self.n_classes,pretrained=self.pretrained)


    def forward(self, X):
        import matplotlib.pyplot as plt
        # print('ResnetClassifierV3', [(key , X[key].shape, X[key].device ) for  key in X.keys()])

        mobilenet_input = X['image']
        # print('resnet_input', resnet_input.shape)

        output = self.model(mobilenet_input)

        return output


class ResNetClassifierV4(nn.Module):
    def __init__(self, 
                model_cfg,
                input_format,
                output_format,
                classes, 
                architecture=None,
                ):
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

        self.input_format = input_format

        self.fcn_input_format = [ inp for inp in self.input_format if inp != 'image']
        self.input_format = [inp for inp in self.input_format if inp == 'image']

        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[self.resnet_version](pretrained=self.pretrained)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, self.n_classes)

        print('linear_size', linear_size)

        if self.tune_fc_only: # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        import matplotlib.pyplot as plt
        # print('ResnetClassifierV3', [(key , X[key].shape, X[key].device ) for  key in X.keys()])

        resnet_input = X['image']
        # print('resnet_input', resnet_input.shape)

        resnet_output = self.resnet_model(resnet_input)

        return resnet_output


class ResNetClassifierV3(nn.Module):
    def __init__(self, 
                model_cfg,
                input_format,
                output_format,
                classes, 
                architecture=None,
                ):
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

        self.input_format = input_format

        self.fcn_input_format = [ inp for inp in self.input_format if inp != 'image']
        self.input_format = [inp for inp in self.input_format if inp == 'image']

        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[self.resnet_version](pretrained=self.pretrained)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Identity()

        print('linear_size', linear_size)

        initial_fcn_layer = nn.Linear(linear_size + len(self.fcn_input_format),self.architecture['fcn_width'])
        hidden_fcn_layers = [ nn.Sequential(
                                    nn.Linear(self.architecture['fcn_width'],self.architecture['fcn_width']),
                                    nn.ReLU()
                                )
                            for i in range(architecture["fcn_num_hidden"])]
        fcn2class_layer = nn.Linear(self.architecture['fcn_width'],self.n_classes)

        self.classifier_fcn = nn.Sequential(
            initial_fcn_layer,
            nn.ReLU(),
            *hidden_fcn_layers,
            fcn2class_layer
        )

        if self.tune_fc_only: # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        # print('ResnetClassifierV3', [(key , X[key].shape, X[key].device ) for  key in X.keys()])

        resnet_input = X['image'].repeat(1,3,1,1)
        # print('resnet_input', resnet_input.shape)
        resnet_output = self.resnet_model(resnet_input)

        # print('resnet_output.shape', resnet_output.shape)
        # print('resnet_output info', torch.min(resnet_output, dim=1), torch.max(resnet_output, dim=1)) 

        norm_resnet_output = F.normalize(input=resnet_output, dim=1)
        # print('norm_resnet_output.shape', norm_resnet_output.shape)
        # print('norm_resnet_output info', torch.min(norm_resnet_output, dim=1), torch.max(norm_resnet_output, dim=1)) 

        fcn_inputs = []
        for dtype in self.fcn_input_format:

            if dtype == 'angular_excentricity':
                #expressed in degrees
                max_normalization = 90
                tmp = torch.squeeze(X[dtype],2) / max_normalization
                fcn_inputs.append(tmp)
            elif dtype == 'centroid_Lat':
                max_normalization = 90 * np.pi / 180
                tmp = torch.squeeze(X[dtype],2) / max_normalization
                fcn_inputs.append(tmp)

            # fcn_inputs.append(torch.squeeze(X[dtype],2))

        fcn_inputs = torch.cat(fcn_inputs, 1 )
        # print("fcn_inputs: ",fcn_inputs.shape)
        fcn_inputs = torch.cat([fcn_inputs, resnet_output], 1 )
        # print("FULL_fcn_inputs: ",fcn_inputs.shape)

        fc_nx = self.classifier_fcn(fcn_inputs)


        return fc_nx

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