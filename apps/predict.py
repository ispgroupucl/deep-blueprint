import streamlit as st

from pathlib import Path
import pandas as pd
from bioblue.plot import cm
import numpy as np
import matplotlib.pyplot as plt
from hydra.experimental import initialize_config_module as init_hydra, compose
import mlflow
import time

import yaml
import os

st.set_page_config(layout="wide")

with init_hydra(config_module="bioblue.conf"):
    cfg = compose(config_name="config", return_hydra_config=True)
    os.environ.update(cfg.hydra.job.env_set)

models_path = Path("../mlruns")
mlflow.set_tracking_uri(cfg.logger.tracking_uri)
exps = [(exp.experiment_id, exp.name) for exp in mlflow.list_experiments()]
chosen_exps = st.sidebar.multiselect("Experiment", exps, format_func=lambda x: x[1])
exp_ids = [exp[0] for exp in chosen_exps]
runs = mlflow.search_runs(experiment_ids=exp_ids)
show_all = st.sidebar.checkbox("Show all", value=True)
if show_all:
    run_columns = runs.columns
else:
    run_columns = st.sidebar.multiselect(
        "show columns", runs.columns, default=["run_id", "metrics.val_loss"]
    )

st.write(
    runs[run_columns]
    .style.bar(
        subset=[c for c in run_columns if "metric" in c], color=["#d65f5f", "#5fba7d"]
    )
    .highlight_null(null_color="white")
)


def get_runname(run_id):
    run = runs.loc[runs["run_id"] == run_id]  # [run_id]
    if "tags.mlflow.runName" in run:
        run_name = run["tags.mlflow.runName"].item()
        return run_name or run_id
    else:
        return run_id


run_id = st.sidebar.selectbox("Run", runs["run_id"], format_func=get_runname)

run = mlflow.get_run(run_id)
run_dir = models_path / run.info.experiment_id / run.info.run_id

for elem in (run_dir / "artifacts").iterdir():
    elem

# time.sleep(10)
# st.experimental_rerun()
# models_paths = [Path("../outputs/"), Path("../multirun")]
# exp_dirs = {}
# for exp_dir in sorted(models_path.iterdir()):
#     meta_file = exp_dir / "meta.yaml"
#     if meta_file.exists():
#         with meta_file.open() as mf:
#             exp_name = yaml.load(mf)["name"]
#         exp_dirs[exp_name] = exp_dir

# chosen_exp_dirs = st.sidebar.multiselect("Experiment", exp_dirs)

# for exp_name in chosen_exp_dirs:
#     exp_dir = exp_dirs[exp_name]
#     for run_dir in exp_dir.iterdir():
#         meta_file = run_dir / "meta.yaml"
#         if meta_file.exists():
#             with meta_file.open() as mf:
#                 meta = yaml.load(mf)
#             meta
