import pytest
from hypothesis import given
import hypothesis.strategies as hs
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hydra import initialize_config_module as init_hydra, compose
from omegaconf import OmegaConf

import bioblue as bb


def test_default_config():
    with init_hydra(config_module="bioblue.conf"):
        cfg = compose(config_name="config", return_hydra_config=False)
        print(OmegaConf.to_yaml(cfg))
