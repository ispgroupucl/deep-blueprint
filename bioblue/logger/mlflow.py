import logging
from bioblue.callback.mlflow import MLFlowCallback
from typing import Any, Dict, Optional
from pytorch_lightning.loggers import MLFlowLogger as BaseLogger
import mlflow
from mlflow.tracking import MlflowClient
from pytorch_lightning.loggers.base import rank_zero_experiment

log = logging.getLogger(__name__)


class MLFlowLogger(BaseLogger):
    def __init__(
        self,
        experiment_name: str = "default",
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = "./mlruns",
        prefix: str = "",
        parent_run=None,
    ):
        super().__init__(experiment_name, tracking_uri, tags, save_dir, prefix)
        self.parent_run = parent_run

    @property
    @rank_zero_experiment
    def experiment(self) -> MlflowClient:
        if self._experiment_id is None:
            expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            if expt is not None:
                self.experiment_id = expt.experiment_id
            else:
                log.warning(
                    f"Experiment with name {self._experiment_name} not found. Creating it."
                )
                self._experiment_id = self._mlflow_client.create_experiment(
                    name=self._experiment_name
                )

        if self._run_id is None:
            run = self._mlflow_client.create_run(
                experiment_id=self._experiment_id, tags=self.tags
            )

