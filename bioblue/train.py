from typing import Mapping
from bioblue.utils.gpu import pick_gpu
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import bioblue.conf
from bioblue.utils import pick_gpu
from hydra.utils import call, instantiate
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    datamodule = instantiate(cfg.dataset, _recursive_=False)
    module = instantiate(cfg.module, _recursive_=False)
    callbacks = []
    if isinstance(cfg.callbacks, Mapping):
        cfg.callbacks = [cb for cb in cfg.callbacks.values()]
    for callback in cfg.callbacks:
        callback = instantiate(callback, _recursive_=False)
        callback.cfg = cfg  # FIXME : ugly hack
        callbacks.append(callback)

    if isinstance(cfg.trainer.gpus, int):
        cfg.trainer.gpus = pick_gpu(cfg.trainer.gpus)

    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=cfg.logger,
        default_root_dir=".",
        callbacks=callbacks,
        _recursive_=True,
        _convert_="all",
    )
    trainer.tune(module, datamodule=datamodule)
    trainer.fit(module, datamodule=datamodule)
    return {key: value.item() for key, value in trainer.callback_metrics.items()}


if __name__ == "__main__":
    main()
