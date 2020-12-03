import pytorch_lightning as pl
from mlflow.tracking import MlflowClient
from pathlib import Path


class MLFlowCallback(pl.Callback):
    def __init__(self, log_models=True, cfg=None) -> None:
        super().__init__()
        self.log_models = log_models
        self.cfg = cfg

    def on_init_end(self, trainer):
        if not isinstance(trainer.logger.experiment, MlflowClient):
            return

        trainer.logger.experiment.log_artifacts(
            trainer.logger.run_id, local_dir="./.hydra", artifact_path="config"
        )
        trainer.logger.log_hyperparams(dict(dataset=self.cfg["dataset"]))

    def on_train_end(self, trainer, pl_module):
        if not isinstance(pl_module.logger.experiment, MlflowClient):
            return

        if Path("./models").exists() and self.log_models:
            pl_module.logger.experiment.log_artifacts(
                run_id=pl_module.logger.run_id,
                local_dir="./models",
                artifact_path="models",
            )
