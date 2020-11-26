"""Functional interface to several data augmentation functions."""
import torch


def random_erase(image, n_blocks=1, max_size=None, return_mask=False):
    """ Erase blocks from the image and replace them by zero.

    Args:
        image (:class:`torch.tensor`): 3D image
        n_blocks (int): number of blocks to remove
        max_size (int or tuple(int)): maximum size of the blocks, if not set 
            will use the max shape divided by 4. This uses a normal distribution
            and so it divides max_size by 4 in order to be more or less always
            smaller than max_size.
        return_mask (bool): if true will return a mask and not the erasure of the
            image
    Returns:
        image or mask : 3D image with random blocks removed
    
    """
    shape = torch.tensor(image.shape)
    mask = torch.ones_like(image)
    if max_size is None:
        max_size = shape / 4.0

    for _ in range(n_blocks):
        center = torch.rand((3,)) * shape
        if type(max_size) is float:
            box = torch.abs(
                torch.normal(mean=0.0, std=max_size / 4.0, size=(3,)).to(torch.int)
            )
        else:
            box = torch.abs(torch.normal(mean=0.0, std=max_size / 4.0).to(torch.int))

        x0, y0, z0 = torch.floor(center - box).to(int)
        x1, y1, z1 = torch.ceil(center + box).to(int)
        x0, x1 = torch.clamp(torch.tensor((x0, x1)), 1, shape[0] - 1)
        y0, y1 = torch.clamp(torch.tensor((y0, y1)), 1, shape[1] - 1)
        z0, z1 = torch.clamp(torch.tensor((z0, z1)), 1, shape[2] - 1)
        mask[x0:x1, y0:y1, z0:z1] = 0

    if return_mask:
        image[mask == 0] = 0
        return image, mask == 0
    else:
        image[mask == 0] = 0
        return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    img = torch.ones((512, 512, 3))
    img = random_erase(img, n_blocks=10, max_size=32)
    plt.imshow(img)
    plt.colorbar()
    plt.show()
