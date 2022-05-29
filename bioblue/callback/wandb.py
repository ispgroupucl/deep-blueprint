import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path


class WandBCallback(pl.Callback):
    def __init__(self, log_models=True, cfg=None) -> None:
        super().__init__()
        self.log_models = log_models
        self.cfg = cfg

    def on_init_end(self, trainer):
        logger:WandbLogger = trainer.logger

        if logger is None:
            return

        logger.log_hyperparams(dict(dataset=self.cfg["dataset"]))

        # # logger.experiment.log_artifacts(
        # #     logger.run_id, local_dir="./.hydra", ar tifact_path="config"
        # )
        # logger.log_hyperparams(dict(dataset=self.cfg["dataset"]))

    def on_train_end(self, trainer, pl_module):
        logger:WandbLogger = pl_module.logger
        if logger is None:
            return

        if Path("./models").exists() and self.log_models:
            artifact = wandb.Artifact('models', type='dataset')
            artifact.add_dir('./models')
            logger.experiment.log_artifact(artifact)

        logger.experiment.finish()

        # if Path("./models").exists() and self.log_models:
        #     logger.experiment.log_artifacts(
        #         run_id=logger.run_id, local_dir="./models", artifact_path="models",
        #     )
