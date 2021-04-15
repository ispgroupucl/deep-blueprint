from typing import List, Union
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from . import Strategy

log = logging.getLogger(__name__)


class VisualizeStrategy(Strategy):
    def __init__(self, visualize=None, train=1, val=1, test=1):
        self.visualize = visualize
        self.train = train
        self.val = val
        self.test = test
        self.vis_numbers = [train, val, test]

    def setup(self, train_ds, val_ds, test_ds) -> None:
        datasets = (train_ds, val_ds, test_ds)
        names = ["train", "val", "test"]
        for i, dataset in enumerate(datasets):
            for idx in range(self.vis_numbers[i]):
                sample = dataset[idx]
                visualize = self.visualize or list(sample.keys())
                fig, axs = plt.subplots(
                    1,
                    ncols=len(visualize),
                    figsize=(10 * len(visualize), 10),
                    squeeze=False,
                )
                for key, ax in zip(visualize, axs[0]):
                    im = ax.imshow(sample[key])
                    fig.colorbar(im, ax=ax)
                directory = Path("images/dataset")
                directory.mkdir(exist_ok=True, parents=True)
                fig.savefig(directory / f"{names[i]}_{idx}.png")
                plt.close(fig)

            if hasattr(dataset, "reset"):
                dataset.reset()
