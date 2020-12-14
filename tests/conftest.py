import os

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://betelgeuse.elen.ucl.ac.be:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://betelgeuse.elen.ucl.ac.be:5000"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
os.environ["AWS_ACCESS_KEY_ID"] = "vjoosdtb"
os.environ["AWS_SECRET_ACCESS_KEY"] = "qsdfghjk"


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

