__version__ = "0.1.0"

from bioblue import dataset, transforms, model, plot
from bioblue import conf
from bioblue.utils import git_info
from omegaconf import OmegaConf
import platform

OmegaConf.register_resolver("host", lambda: platform.node().split(".", 1)[0].lower())
OmegaConf.register_resolver("uname", lambda x: getattr(platform.uname(), x))
OmegaConf.register_resolver("git", git_info)


__all__ = ["dataset", "transforms", "model", "conf", "plot"]
