import os
from hydra import initialize_config_module as init_hydra, compose

with init_hydra(config_module="bioblue.conf"):
    cfg = compose(config_name="config", return_hydra_config=True)
    os.environ.update(cfg.hydra.job.env_set)


def pytest_addoption(parser):
    parser.addoption(
        "--longrun",
        action="store_true",
        dest="longrun",
        default=False,
        help="enable long undecorated tests",
    )


def pytest_configure(config):
    if not config.option.longrun:
        setattr(config.option, "markexpr", "not slow")

