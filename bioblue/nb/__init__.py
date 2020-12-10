# Default imports for notebook
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.notebook import tqdm
from itkwidgets import view
import cv2
import ipywidgets as widgets
import os

from mlflow.tracking import MlflowClient
import mlflow

from hydra.experimental import initialize_config_module as init_hydra, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import bioblue as bb

from bioblue.nb.image_display import MultiImageDisplay


# Initialize environment variables
with init_hydra(config_module="bioblue.conf"):
    cfg = compose(config_name="config", return_hydra_config=True)
    os.environ.update(cfg.hydra.job.env_set)

mlflow.set_tracking_uri(cfg.logger.tracking_uri)
mlflow_client = MlflowClient()
