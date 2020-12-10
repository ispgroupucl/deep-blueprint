__version__ = "0.1.0"

from bioblue import dataset, transforms, model

from omegaconf import OmegaConf
import platform

OmegaConf.register_resolver("host", lambda: platform.node().split(".", 1)[0])
OmegaConf.register_resolver("uname", lambda x: getattr(platform.uname(), x))

__all__ = [
    "dataset",
    "transforms",
    "model",
]
