import streamlit as st

from pathlib import Path
import pandas as pd
from bioblue.plot import cm
import numpy as np
import matplotlib.pyplot as plt
from hydra.experimental import initialize_config_module as init_hydra, compose
from omegaconf import OmegaConf
import mlflow
import time
import cv2

import yaml
import os

st.set_page_config(layout="wide")


def pathname(x):
    return x.name


with init_hydra(config_module="bioblue.conf"):
    cfg = compose(config_name="config", return_hydra_config=True)
    os.environ.update(cfg.hydra.job.env_set)

number = st.sidebar.slider("Number of models", min_value=1, max_value=5)

sb_container = st.sidebar.beta_container()
main_container = st.beta_container()
i, mask_dir = None, None

for model_i in range(number):
    st.sidebar.write(f"### Model {model_i}")
    runs_path = Path("../outputs/")
    day_path = st.sidebar.selectbox(
        "Day",
        sorted(runs_path.iterdir(), reverse=True),
        format_func=pathname,
        key=f"day_{model_i}",
    )
    run_path = st.sidebar.selectbox(
        "Run", sorted(day_path.iterdir()), format_func=pathname, key=f"run_{model_i}"
    )
    st.write(f"## Model {run_path.parent.name} {run_path.name.replace('-',':')}")

    image_path = run_path / "images/volumes"
    config_path = run_path / ".hydra/config.yaml"

    config = OmegaConf.load(config_path)
    ds_conf = config.dataset  # OmegaConf.to_container(config.dataset, resolve=True)
    if model_i == 0:
        image_epoch = sb_container.selectbox(
            "Epoch",
            sorted(image_path.iterdir()),
            format_func=pathname,
            key=f"image_epoch_{model_i}",
        )
        sample = sb_container.selectbox(
            "Sample",
            sorted(image_epoch.iterdir()),
            format_func=pathname,
            key=f"sample_{model_i}",
        )
    sample = image_path / image_epoch.name / sample.name
    bg_image_path = (
        Path(ds_conf.data_dir)
        / ds_conf.dataset_name
        / "train"
        / "image"
        / f"{sample.name}.npz"
    )
    seg_path = (
        Path(ds_conf.data_dir)
        / ds_conf.dataset_name
        / "train"
        / "segmentation"
        / f"{sample.name}.npz"
    )
    image = np.load(bg_image_path)
    gt_seg = np.load(seg_path)
    if model_i == 0:
        i, mask_dir = main_container.select_slider(
            "Image",
            enumerate(sorted(sample.iterdir())),
            format_func=lambda x: x[0],
            key=f"image_{model_i}",
        )
    # st.write(image_path, mask_dir)
    mask_dir = sorted(sample.iterdir())[i]
    idx = int(mask_dir.stem)
    img = image[image.files[idx]]
    col1, col2 = st.beta_columns(2)
    mask = cv2.imread(str(mask_dir), cv2.IMREAD_UNCHANGED)
    gt = gt_seg[gt_seg.files[idx]]
    fig = plt.figure()
    plt.imshow(img)
    plt.imshow(gt, cmap=cm.rb, alpha=0.6, interpolation="none")
    # plt.colorbar()
    col1.pyplot(fig)

    fig = plt.figure()
    plt.imshow(img)
    # mask[mask == 2] = 0
    plt.imshow(mask, cmap=cm.rb, alpha=0.5, interpolation="none", vmin=0)
    # plt.colorbar()
    col2.pyplot(fig)
