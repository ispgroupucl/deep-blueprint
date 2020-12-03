import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import bioblue.conf
from hydra.utils import call, instantiate
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.dataset)
    module = instantiate(cfg.module)
    logger = instantiate(cfg.logger)
    callbacks = []
    for callback in cfg.callbacks:
        callback = instantiate(callback)
        callback.cfg = cfg  # FIXME : ugly hack
        callbacks.append(callback)
    trainer = instantiate(
        cfg.trainer, logger=logger, default_root_dir=".", callbacks=callbacks
    )
    trainer.tune(module, datamodule=datamodule)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
