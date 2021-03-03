import numpy as np


def transform(f, sample, *args, **kwargs):
    state = np.random.get_state()
    data, seg = f(data=sample["image"], seg=sample["segmentation"], *args, **kwargs)
    sample["image"], sample["segmentation"] = data, seg
    for name, item in sample:
        if name.startswith("image_"):
            np.random.set_state(state)
            f(data=item, *args, **kwargs)
