import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

import os
import glob


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
            cwd = os.getcwd()
            print(cwd)
            models_lst = sorted(glob.glob(os.path.join('./models', '*.ckpt')))
            for model in models_lst:
                if os.path.basename(model) == 'last.ckpt':
                    artifact.add_file(model, os.path.join("models", os.path.basename(model)))

            logger.experiment.log_artifact(artifact)

        logger.experiment.finish()

        # if Path("./models").exists() and self.log_models:
        #     logger.experiment.log_artifacts(
        #         run_id=logger.run_id, local_dir="./models", artifact_path="models",
        #     )

    def on_test_end(self, trainer, pl_module):
        logger:WandbLogger = pl_module.logger
        if logger is None:
            return
         
        logger.experiment.finish()

