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
    # trainer = hydra.utils.instantiate(cfg.trainer)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    # print(cfg.module.segmenter, module)
    trainer = pl.Trainer()
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
