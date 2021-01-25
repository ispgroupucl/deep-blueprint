from typing import List, Union
import numpy as np


def pick_gpu(number: Union[int, List]):
    from subprocess import check_output

    if isinstance(number, List):
        return number

    gpu_stats = [
        int(x.split(", ")[0]) / int(x.split(", ")[1])
        for x in check_output(
            [
                "nvidia-smi",
                "--format=csv,nounits,noheader",
                "--query-gpu=memory.used,memory.total",
            ]
        )
        .decode()
        .strip()
        .split("\n")
    ]
    if len(gpu_stats) < number:
        raise ValueError(f"Not enough GPU's for the requested {number}.")

    return [int(x) for x in np.argsort(gpu_stats)[:number]]


if __name__ == "__main__":
    print(pick_gpu(0))
