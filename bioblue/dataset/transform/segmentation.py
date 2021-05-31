from albumentations import DualTransform


class BinarySegmentation(DualTransform):
    def __init__(self, keep: 1):
        self.keep = keep
        super().__init__(always_apply=True)

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **kwargs):
        mask[mask != self.keep] = 0
        mask[mask == self.keep] = 1
        return mask

