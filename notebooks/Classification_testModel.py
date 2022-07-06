# %%
import wandb
# import util
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import shutil
import wandb
from PIL import Image
import os
import glob
from pathlib import Path

from tqdm.notebook import tqdm

from omegaconf import OmegaConf



from bioblue.nb.load import load_from_cfg

# %%


# %%
config = OmegaConf.load('/home/ucl/elen/nsayez/bio-blueprints/bioblue/conf/exp/DeepsunClassification.yaml')
config['use_dtypes'] = ["image","T400-T350-Alternating"]
print(config)


# %%
module, datamodule = load_from_cfg(config, recursive=False)
# datamodule.train_ds
datamodule.setup()
train_dl = datamodule.train_dataloader()
next(iter(train_dl))


# train_dl = datamodule.train_dataloader()
# next(iter(train_dl))

# %%
# {'seed': 0, 'trainer': {'_target_': 'pytorch_lightning.Trainer', 'accumulate_grad_batches': None, 'amp_backend': 'native', 'amp_level': None, 'log_gpu_memory': None, 'auto_lr_find': False, 'auto_scale_batch_size': False, 'auto_select_gpus': False, 'benchmark': None, 'enable_checkpointing': True, 'check_val_every_n_epoch': 1, 'detect_anomaly': False, 'deterministic': False, 'fast_dev_run': False, 'gpus': '${gpus}', 'gradient_clip_val': None, 'gradient_clip_algorithm': None, 'limit_train_batches': None, 'limit_val_batches': None, 'limit_test_batches': None, 'limit_predict_batches': None, 'log_every_n_steps': 2, 'enable_progress_bar': True, 'profiler': None, 'overfit_batches': 0.0, 'plugins': None, 'precision': 32, 'max_epochs': 30, 'min_epochs': None, 'max_steps': -1, 'min_steps': None, 'max_time': None, 'num_nodes': 1, 'num_processes': None, 'num_sanity_val_steps': 2, 'reload_dataloaders_every_n_epochs': 0, 'strategy': None, 'sync_batchnorm': False, 'track_grad_norm': -1, 'val_check_interval': 1.0, 'enable_model_summary': True, 'weights_save_path': None, 'move_metrics_to_cpu': False, 'multiple_trainloader_mode': 'max_size_cycle'}, 'model': {'_target_': 'bioblue.model.ConfResNet', 'model_cfg': {'_target_': 'bioblue.model.ModelConfig'}, 'architecture': {'first': 32, 'enc': {'width': [16, 32, 48, 96], 'repeat': [2, 3, 3, 4]}, 'dec': {'width': [48, 32, 32], 'repeat': [2, 2, 1]}, 'block_width': [32, 64, 128], 'num_resnet_blocks': [6, 6, 6]}, 'input_format': ['image'], 'output_format': ['class'], 'classes': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'Unknown']}, 'module': {'_target_': 'bioblue.module.BaseClassifier', 'segmenter': '${model}', 'lr': 0.0001, 'classifier': '${model}', 'optimizer': 'torch.optim.AdamW', 'loss': {'_target_': 'torch.nn.CrossEntropyLoss'}, 'scheduler': {'_target_': 'torch.optim.lr_scheduler.StepLR', 'step_size': 250, 'gamma': 0.9, 'last_epoch': -1}}, 'dataset': {'_target_': 'bioblue.dataset.BioblueDataModule', 'data_dir': '/globalscratch/users/n/s/nsayez/Classification_dataset/', 'directory': 'synthetic', 'train_size': 50, 'val_size': 50, 'test_size': 50, 'shape': [512, 512], 'batch_size': 16, 'points_range': [20, 21], 'links_range': [3, 4], 'max_shapes': 1000, 'size_range': [50, 500], 'weight_range': [1, 5], 'bg_intensity_range': [0, 50], 'fg_intensity_range': [0, 256], 'num_workers': 12, 'dataset_name': '2002-2019', 'train_dataset': {'_target_': 'bioblue.dataset.DeepsunMaskedClassificationDataset', 'root_dir': '${dataset.data_dir}', 'partition': 'train', 'dtypes': '${use_dtypes}'}, 'val_dataset': {'_target_': 'bioblue.dataset.DeepsunSegmentationDataset', 'partition': 'test', 'dtypes': '${use_dtypes}'}, 'test_dataset': {'_target_': 'bioblue.dataset.DeepsunSegmentationTestDataset', 'partition': 'test_GT', 'dtypes': ['image', 'GroundTruth']}}, 'logger': [], 'exp': '???', 'gpus': 1, 'use_dtypes': ['image', 'T400-T350-Alternating']}

# %%



