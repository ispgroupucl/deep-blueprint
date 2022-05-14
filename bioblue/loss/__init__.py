# from .dice import dice_loss, DiceLoss
from .soft_dice import SoftDiceLoss
from torch.nn import CrossEntropyLoss
from monai.losses import  GeneralizedDiceLoss
from .loss_segmentation import TverskyLossV2 ,TverskyLossV3
