import numpy as np

from copy import deepcopy
from albumentations.core.transforms_interface import DualTransform
from torchvision.transforms import Grayscale

class RepeatSingleChannel(DualTransform):
    def __init__(self, always_apply=False, num_repeat=3):
        super().__init__(always_apply)
        self.repeater = Grayscale(num_output_channels=num_repeat)


    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunRotateAndCropAroundGroup')
        img = kwargs['image'].copy()
        img = np.repeat(img[None,:,:],3,axis=0)
        kwargs['image'] = img 

        return kwargs