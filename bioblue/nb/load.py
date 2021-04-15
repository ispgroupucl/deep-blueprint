from bioblue.utils.gpu import pick_gpu
from typing import Any, Dict, Mapping, Optional, Tuple
from hydra.utils import instantiate
from hydra.experimental import initialize_config_module as init_hydra, compose
from hydra.core.utils import configure_log
from mlflow.tracking.client import MlflowClient
from omegaconf import OmegaConf, DictConfig
import mlflow
import pytorch_lightning as pl
from tempfile import TemporaryDirectory
from importlib import import_module


def load_from_runid(run_id: str, ckpt_name: str = "epoch"):
    mlflow_client = MlflowClient()
    run = mlflow.get_run(run_id)

    with TemporaryDirectory() as directory:
        configfile = mlflow_client.download_artifacts(
            run_id, "config/config.yaml", directory
        )
        modelfile = None
        for modelinfo in mlflow_client.list_artifacts(run_id, path="models"):
            if ckpt_name in modelinfo.path:
                modelfile = mlflow_client.download_artifacts(
                    run_id, modelinfo.path, dst_path=directory
                )
                break
        if modelfile is None:
            raise ValueError(
                f"Either ckpt_name ({ckpt_name}) is invalid or another error occured."
            )
        cfg = OmegaConf.load(configfile)
        target_module, target_class = cfg.module["_target_"].rsplit(".", 1)
        target_module = import_module(target_module)
        target_class = getattr(target_module, target_class)
        module = target_class.load_from_checkpoint(modelfile)
        datamodule = instantiate(cfg.dataset)
        datamodule.prepare_data()
        datamodule.setup()
    return cfg, module, datamodule


def load_from_cfg(cfg: DictConfig) -> Tuple[pl.LightningModule, pl.LightningDataModule]:
    datamodule = instantiate(cfg.dataset)
    module = instantiate(cfg.module)

    return module, datamodule


def load_from_overrides(overrides=[], load_trainer=False) -> Tuple:
    with init_hydra(config_module="bioblue.conf"):
        cfg = compose(
            config_name="config", overrides=overrides, return_hydra_config=True
        )

    configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    datamodule = instantiate(cfg.dataset)
    module = instantiate(cfg.module)
    trainer: Optional[pl.Trainer] = None
    if load_trainer:
        logger = instantiate(cfg.logger)
        callbacks = []
        if isinstance(cfg.callbacks, Mapping):
            cfg.callbacks = [cb for cb in cfg.callbacks.values()]
        for callback in cfg.callbacks:
            callback = instantiate(callback)
            callback.cfg = cfg  # FIXME : ugly hack
            callbacks.append(callback)

        if isinstance(cfg.trainer.gpus, int):
            cfg.trainer.gpus = pick_gpu(cfg.trainer.gpus)

        trainer: pl.Trainer = instantiate(
            cfg.trainer, logger=logger, default_root_dir=".", callbacks=callbacks
        )

    return cfg, module, datamodule, trainer
