from dataclasses import dataclass, field
import os
import subprocess
import multiprocessing as mp
from multiprocessing.connection import wait
import logging
from typing import List
import cloudpickle
import dill
import sys
from pathlib import Path
from mlflow import tracking
from pytorch_lightning.loggers import MLFlowLogger
from hydra.core.config_loader import ConfigLoader
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.plugins.launcher import Launcher
from jinja2.loaders import PackageLoader
from jinja2.runtime import StrictUndefined
from jinja2.utils import select_autoescape
from mlflow.tracking.client import MlflowClient
from omegaconf import DictConfig, open_dict
from argparse import ArgumentParser
from jinja2 import Environment

from hydra.core.utils import (
    JobReturn,
    configure_log,
    env_override,
    filter_overrides,
    run_job,
    setup_globals,
)


log = logging.getLogger(__name__)


@dataclass
class SlurmLauncherConfig:
    _target_: str = "hydra_plugins.bioblue_launcher.launcher.SlurmLauncher"
    jobname: str = "${exp}"
    time: str = "5-00:00:00"
    cpus_per_task: int = 8
    mem: str = "16GB"
    gpu_name: str = ""
    gpus: int = 1
    partition: str = "gpu"
    additional_sbatch: list = field(default_factory=list)


@dataclass
class GPULauncherConfig:
    _target_: str = "hydra_plugins.bioblue_launcher.launcher.GPULauncher"
    gpus: List[int] = field(default_factory=lambda: [0])


ConfigStore.instance().store(
    group="hydra/launcher", name="slurm", node=SlurmLauncherConfig
)
ConfigStore.instance().store(group="hydra/launcher", name="gpu", node=GPULauncherConfig)


class BioblueLauncher(Launcher):
    def setup(self, config, config_loader, task_function):
        self.config = config
        self.config_loader = config_loader
        self.task_function = task_function
        self.run_id = None
        if "experiment_name" in self.config.logger:
            with env_override(self.config.hydra.job.env_set):
                mlflow_client = MlflowClient()
                exp_name = self.config.logger.experiment_name
                tracking_uri = self.config.logger.tracking_uri
                tags = self.config.logger.tags
                self.mlflow_logger = MLFlowLogger(
                    exp_name, tracking_uri=tracking_uri, tags=tags
                )
                self.run_id = self.mlflow_logger.run_id


class SlurmLauncher(BioblueLauncher):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = None
        self.config_loader = None
        self.task_function = None
        self.kwargs = kwargs

    def setup(self, config, config_loader, task_function):
        super().setup(config, config_loader, task_function)

    def launch(self, job_overrides, initial_job_idx):
        setup_globals()

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        log.debug(sweep_dir)
        (sweep_dir / ".slurm").mkdir(parents=True, exist_ok=True)
        commands = []
        output_files = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            sweep_config = self.config_loader.load_sweep_config(
                self.config, list(overrides)
            )
            job_params = {
                "class": self,
                "params": {
                    "sweep_overrides": overrides,
                    "job_num": idx,
                    "job_id": f"job_id_for_{idx}",
                    "singleton_state": Singleton.get_state(),
                },
            }
            with open(sweep_dir / ".slurm" / f"input_{idx}.pkl", "wb") as pkl_file:
                dill.dump(
                    job_params, pkl_file, protocol=dill.HIGHEST_PROTOCOL, recurse=True,
                )
            input_file = sweep_dir / ".slurm" / f"input_{idx}.pkl"
            output_file = sweep_dir / ".slurm" / f"output_{idx}.pkl"
            env = Environment(
                loader=PackageLoader("hydra_plugins.bioblue_launcher", "templates"),
                # undefined=StrictUndefined,
                autoescape=select_autoescape(["html", "xml"]),
            )

            run_file = sweep_dir / ".slurm" / f"run_{idx}.sh"

            with open(run_file, "w") as f:
                rendered = env.get_template("run.sh.jinja").render(
                    input_file=input_file,
                    output_file=output_file,
                    python_command=sys.executable,
                    **self.kwargs,
                )
                f.write(rendered)

            commands.append(subprocess.Popen(["bash", run_file]))
            output_files.append(output_file)

        rets = []
        for p, output_file in zip(commands, output_files):
            p.wait()
            with open(output_file, "rb") as f:
                ret = dill.load(f)
                if isinstance(ret, Exception):
                    raise ret
                rets.append(ret)

        print(rets)
        return rets

    def __call__(self, sweep_overrides, job_num, job_id, singleton_state):
        Singleton.set_state(singleton_state)
        setup_globals()

        sweep_config = self.config_loader.load_sweep_config(
            self.config, sweep_overrides
        )

        with open_dict(sweep_config.hydra.job) as job:
            job.id = job_id
            job.num = job_num
            if self.run_id is not None:
                sweep_config.logger.tags["mlflow.parentRunId"] = self.run_id

        return run_job(
            config=sweep_config,
            task_function=self.task_function,
            job_dir_key="hydra.sweep.dir",
            job_subdir_key="hydra.sweep.subdir",
        )


class GPULauncher(BioblueLauncher):
    def __init__(self, gpus=(0,)) -> None:
        self.gpus = list(gpus)

    def launch(self, job_overrides, initial_job_idx):
        setup_globals()
        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = self.config.hydra.sweep.dir
        Path(str(sweep_dir)).mkdir(parents=True, exist_ok=True)
        sentinels = {}
        processes = {}
        runs = []
        self.mlflow_logger.log_hyperparams(self.config.module)
        self.mlflow_logger.log_hyperparams(dict(dataset=self.config.dataset))
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            if len(processes) == len(self.gpus):
                finished_sentinels = wait(list(sentinels.keys()))
                for sentinel in finished_sentinels:
                    gpu = sentinels[sentinel]
                    job_idx, p, pipe_recv = processes[gpu]
                    p.join()
                    ret: JobReturn = pipe_recv.recv()
                    log.info(f"job #{job_idx} returned {ret.return_value}")
                    runs.append(ret)
                    del processes[gpu]
                    del sentinels[sentinel]
            sweep_config = self.config_loader.load_sweep_config(self.config, overrides)

            available_gpu = next(gpu for gpu in self.gpus if gpu not in processes)
            with open_dict(sweep_config):
                sweep_config.hydra.job.id = idx
                sweep_config.hydra.job.num = idx
                sweep_config.gpus = [available_gpu]

                if self.run_id is not None:
                    sweep_config.logger.tags["mlflow.parentRunId"] = self.run_id

            HydraConfig.instance().set_config(sweep_config)
            lst = " ".join(filter_overrides(overrides))
            log.info(f"Running job #{idx} ({lst}) on gpu:{available_gpu}")
            pipe_recv, pipe_send = mp.Pipe(duplex=False)
            process = mp.Process(
                target=self,
                kwargs=dict(
                    config=sweep_config,
                    task_function=self.task_function,
                    job_dir_key="hydra.sweep.dir",
                    job_subdir_key="hydra.sweep.subdir",
                    pipe_send=pipe_send,
                ),
            )
            process.start()
            for gpu in self.gpus:
                if gpu not in processes:
                    processes[gpu] = (idx, process, pipe_recv)
                    sentinels[process.sentinel] = gpu
                    break
            # runs.append(ret)
            # configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        for _, process in processes.items():
            job_idx, proc, pipe_recv = process
            proc.join()
            ret: JobReturn = pipe_recv.recv()
            log.info(f"job #{job_idx} returned {ret.return_value}")
            runs.append(ret)

        metric_means = {}
        for key in runs[0].return_value:
            metric_means[key] = sum(run.return_value[key] for run in runs) / len(runs)

        self.mlflow_logger.log_metrics(metric_means)
        self.mlflow_logger.finalize()
        return runs

    def __call__(self, config, task_function, job_dir_key, job_subdir_key, pipe_send):
        ret = run_job(
            config=config,
            task_function=task_function,
            job_dir_key=job_dir_key,
            job_subdir_key=job_subdir_key,
        )
        pipe_send.send(ret)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()
    with open(args.input_file, "rb") as f:
        params = dill.load(f)
    load_class = params["class"]
    try:
        ret = load_class(**params["params"])
    except Exception as e:
        ret = e
    with open(args.output_file, "wb") as f:
        dill.dump(ret, f)
    print(args.input_file, args.output_file)
