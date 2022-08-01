import pytorch_lightning as pl
from mlflow.tracking import MlflowClient
from pathlib import Path


class MLFlowCallback(pl.Callback):
    def __init__(self, log_models=True, cfg=None) -> None:
        super().__init__()
        self.log_models = log_models
        self.cfg = cfg

    def get_mlflowclient(self, logger):
        if isinstance(logger.experiment, MlflowClient):
            return logger
        elif isinstance(logger.experiment, list):
            for i, exp in enumerate(logger.experiment):
                if isinstance(exp, MlflowClient):
                    return logger[i]
        return None

    def on_init_end(self, trainer):
        logger = self.get_mlflowclient(trainer.logger)
        if logger is None:
            return

        logger.experiment.log_artifacts(
            logger.run_id, local_dir="./.hydra", artifact_path="config"
        )
        logger.log_hyperparams(dict(dataset=self.cfg["dataset"]))

    def on_train_end(self, trainer, pl_module):
        logger = self.get_mlflowclient(pl_module.logger)
        if logger is None:
            return

        if Path("./models").exists() and self.log_models:
            logger.experiment.log_artifacts(
                run_id=logger.run_id, local_dir="./models", artifact_path="models",
            )
