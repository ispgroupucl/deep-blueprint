from collections import defaultdict
from tqdm import tqdm
import torch
import pytorch_lightning as pl


def generate_results(model: pl.LightningDataModule, dataset):
    # results = dict(
    #     image=defaultdict(list), seg=defaultdict(list), seg_hat=defaultdict(list)
    # )
    results = defaultdict(lambda: dict(image=[], seg=[], seg_hat=[]))
    for i, sample in enumerate(tqdm(dataset)):
        model_input = {
            "image": torch.from_numpy(sample["image"]).to(torch.float).unsqueeze(0),
            "mip-image": torch.from_numpy(sample["mip-image"])
            .to(torch.float)
            .unsqueeze(0),
        }
        seg_hat = model(model_input)
        filename = dataset.files[dataset.reverse_index["image"][i]]
        results[filename]["image"].append(sample["image"])
        results[filename]["seg"].append(sample["segmentation"])
        results[filename]["seg_hat"].append(seg_hat)

    return results
