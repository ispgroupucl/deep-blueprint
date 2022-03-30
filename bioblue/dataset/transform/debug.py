from matplotlib import pyplot as plt
from itkwidgets import view
from monai.transforms import MapTransform, RandomizableTransform, InvertibleTransform
import numpy as np
import torch


class ShowHisto(MapTransform):
    def __init__(self, keys=["image", "segmentation"], title="") -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.title = title

    def __call__(self, data):
        print(data.keys())
        keys = list(self.key_iterator(data))
        fig, axs = plt.subplots(
            ncols=len(keys), figsize=(10 * len(keys), 10), squeeze=False
        )
        for key, ax in zip(keys, axs[0]):
            d = data[key]  # .cpu().numpy()
            if isinstance(d, torch.Tensor):
                d = d.cpu().numpy()
            dmin, dmax = int(d.min()), int(d.max())
            ax.hist(d.flatten(), bins=dmax - dmin + 1, range=(dmin, dmax))
            ax.set_title(f"histogram {key} {self.title}")
        plt.show()
        return data
