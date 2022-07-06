# Default imports for notebook
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage
from tqdm.auto import tqdm
import k3d
import vtk
from itkwidgets import view, compare
import nibabel as nib
import cv2
import ipywidgets as widgets
import os
from collections import defaultdict
import seaborn as sns
from IPython.display import display, Markdown

from mlflow.tracking import MlflowClient
import mlflow

from hydra import initialize_config_module as init_hydra, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics.utilities.data import to_categorical, to_onehot

import bioblue as bb
import bioblue.plot.cm as cm

from bioblue.nb.image_display import MultiImageDisplay, myshow, myshow3d
from bioblue.nb.load import load_from_cfg, load_from_runid, load_from_overrides, load_from_overrides_and_modelpath

# Initialize environment variables
# with init_hydra(config_module="bioblue.conf"):
#     cfg = compose(config_name="config", return_hydra_config=True)
#     os.environ.update(cfg.hydra.job.env_set)


# for logger in cfg.logger:
#     if hasattr(logger, "tracking_uri"):
#         mlflow.set_tracking_uri(logger.tracking_uri)
#         mlflow_client = MlflowClient()
#         break
