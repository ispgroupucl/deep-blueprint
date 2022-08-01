from ..utils.gpu import pick_gpu
from typing import Any, Dict, Mapping, Optional, Tuple
from hydra.utils import instantiate
from hydra import initialize_config_module as init_hydra, compose, initialize_config_dir
from hydra import initialize
from hydra.core.utils import configure_log
from mlflow.tracking.client import MlflowClient
from omegaconf import ListConfig, OmegaConf, DictConfig
import mlflow
import pytorch_lightning as pl
from tempfile import TemporaryDirectory
from importlib import import_module
from pytorch_lightning import LightningModule
from deep_blueprint import module
from omegaconf import OmegaConf
from pathlib import Path


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


def load_from_cfg(cfg: DictConfig, recursive=True) -> Tuple[pl.LightningModule, pl.LightningDataModule]:
    datamodule = instantiate(cfg.dataset,  _recursive_=recursive)
    module = instantiate(cfg.module, _recursive_=recursive)

    return module, datamodule


def load_from_dir(run_path, model_path=None, load_trainer=False, override=None):
    config_path = run_path / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)
    if override is not None:

        with init_hydra(config_module="deep_blueprint.conf"):
            cfg2 = compose(
                config_name="config", overrides=override, return_hydra_config=True
            )

        cfg = OmegaConf.merge(cfg, cfg2)

    module_class: LightningModule = getattr(
        module, cfg.module._target_.split(".")[-1]
    )
    if model_path is None:
        model_path = run_path / "models/last.ckpt"
    model = module_class.load_from_checkpoint(model_path)


    datamodule = instantiate(cfg.dataset, _recursive_=False)
    datamodule.prepare_data()
    # datamodule.setup()

    trainer: Optional[pl.Trainer] = None
    if load_trainer:
        # print(type(cfg.logger[0]))
        tmp_logger = cfg.logger if type(cfg.logger) == DictConfig else cfg.logger[0]
        logger = instantiate(tmp_logger)
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

    return cfg, model, datamodule, trainer

def load_from_dir2(run_path, model_path=None, load_trainer=False, override=None):
    config_path = run_path / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)
    
    # print(type(cfg))

    with initialize_config_dir(config_dir=str(config_path.parents[0])):
        cfg = compose(
            config_name="config", overrides=override, return_hydra_config=True
        )
    

    module_class: LightningModule = getattr(
        module, cfg.module._target_.split(".")[-1]
    )
    if model_path is None:
        model_path = run_path / "models/last.ckpt"
    model = module_class.load_from_checkpoint(model_path)

    # print(cfg.dataset)

    datamodule = instantiate(cfg.dataset, _recursive_=False)
    datamodule.prepare_data()
    # datamodule.setup()

    trainer: Optional[pl.Trainer] = None
    if load_trainer:
        # print(type(cfg.logger[0]))
        tmp_logger = cfg.logger if type(cfg.logger) == DictConfig else cfg.logger[0]
        logger = instantiate(tmp_logger)
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

    return cfg, model, datamodule, trainer


def load_from_overrides(overrides=[], load_trainer=False) -> Tuple:
    with init_hydra(config_module="deep_blueprint.conf"):
        cfg = compose(
            config_name="config", overrides=overrides, return_hydra_config=True
        )

    configure_log(cfg.hydra.job_logging, cfg.hydra.verbose)
    print(cfg.module)
    datamodule = instantiate(cfg.dataset, _recursive_=False)
    module = instantiate(cfg.module, _recursive_=False)
    trainer: Optional[pl.Trainer] = None
    if load_trainer:
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
            cfg.trainer, logger=cfg.logger, default_root_dir=".", callbacks=callbacks
        )

    return cfg, module, datamodule, trainer



def load_from_overrides_and_modelpath (overrides=[], model_path=None , load_trainer=False) -> Tuple:
    # cfg = OmegaConf.create()
    with init_hydra(config_module="deep_blueprint.conf"):
        cfg = compose(
            config_name="config", overrides=overrides, return_hydra_config=True
        )
        print(type(cfg))

    module_class: LightningModule = getattr(
        module, cfg.module._target_.split(".")[-1]
    )
    if model_path is None:
        model_path = Path('.') / "models/last.ckpt"
    model = module_class.load_from_checkpoint(model_path)


    datamodule = instantiate(cfg.dataset, _recursive_=False)
    datamodule.prepare_data()
    # datamodule.setup()

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

    
    return cfg, model, datamodule, trainer