import hydra
from omegaconf import DictConfig
import torch
import bioblue.conf
from hydra.core.config_store import ConfigStore


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    print(ConfigStore.instance())


if __name__ == "__main__":
    main()
