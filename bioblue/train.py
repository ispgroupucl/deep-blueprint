import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import bioblue.conf
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    datamodule = instantiate(cfg.dataset)
    module = instantiate(cfg.module)
    trainer = instantiate(cfg.trainer)
    trainer.tune(module, datamodule=datamodule)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
