# from .dice import dice_loss, DiceLoss
from .soft_dice import SoftDiceLoss
from torch.nn import CrossEntropyLoss
from monai.losses import  GeneralizedDiceLoss
from .loss_segmentation import DiceLoss
from monai.losses.dice import DiceCELoss
from monai.losses import  GeneralizedDiceLoss
from .Tversky import TverskyLossV2 ,TverskyLossV3
from .combo_loss import ComboLoss, CombineLosses
from .log_cosh_dice import LogCosHDiceLoss
