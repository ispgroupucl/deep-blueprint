import streamlit as st
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import torch
from bioblue import module


@st.cache()
def load_from_dir(run_path):
    config_path = run_path / ".hydra/config.yaml"
    config = OmegaConf.load(config_path)
    module_class: LightningModule = getattr(
        module, config.module._target_.split(".")[-1]
    )
    model_path = run_path / "models/last.ckpt"

    model = module_class.load_from_checkpoint(model_path)
    datamodule = instantiate(config.dataset, _recursive_=False)
    datamodule.prepare_data()
    datamodule.setup()
    return config, model, datamodule


exp = st.sidebar.selectbox(
    "experiment",
    sorted(Path("../multirun").glob("*/*"), reverse=True),
    format_func=lambda x: f"{x.parent.name} {x.name.replace('-',':')}",
)

configs = []
models = []
datamodules = []

for run in sorted(exp.iterdir()):
    if not run.is_dir():
        continue
    config, model, datamodule = load_from_dir(run)
    configs.append(config)
    models.append(model)
    datamodules.append(datamodule)

val_dl = st.cache(datamodules[0].val_dataloader)()
for batch in val_dl:
    segms = []
    for model in models:
        segmentation = model(dict(image=batch["image"])).numpy()
        segms.append(segmentation)
    segms
    # fig = plt.figure()
    # plt.imshow(img)
    # st.pyplot(fig)
    break
