from typing import Optional, Sequence, Union
import torch
import torch.nn.functional as F
from kornia.utils import one_hot
import logging

log = logging.getLogger(__name__)


class SoftDiceLoss(torch.nn.Module):
    """Soft Dice Loss"""

    def __init__(
        self,
        square_nom: bool = False,
        square_denom: bool = False,
        weight: Optional[Union[Sequence, torch.Tensor]] = None,
        smooth: float = 1.0,
    ):
        """
        Args:
            square_nom: whether to square the nominator
            square_denom: whether to square the denominator
            weight: additional weighting of individual classes
            smooth: smoothing for nominator and denominator
    
        """
        super().__init__()
        self.square_nom = square_nom
        self.square_denom = square_denom

        self.smooth = smooth

        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight)

            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes SoftDice Loss
        
        Args:
            predictions: the predictions obtained by the network
            targets: the targets (ground truth) for the :attr:`predictions`
            
        Returns:
            torch.Tensor: the computed loss value
        """
        # number of classes for onehot
        n_classes = predictions.shape[1]
        predictions = F.softmax(predictions, dim=1)
        log.debug(f"{targets.device}, {predictions.device}")
        with torch.no_grad():
            log.debug(f"{targets.device}, {predictions.device}")
            targets_onehot = one_hot(
                targets, num_classes=n_classes, device=predictions.device
            )
        # sum over spatial dimensions
        log.debug(f"{targets_onehot.shape} {predictions.shape}")
        dims = tuple(range(2, predictions.dim()))

        # compute nominator
        if self.square_nom:
            nom = torch.sum((predictions * targets_onehot.float()) ** 2, dim=dims)
        else:
            nom = torch.sum(predictions * targets_onehot.float(), dim=dims)
        nom = 2 * nom + self.smooth

        # compute denominator
        if self.square_denom:
            i_sum = torch.sum(predictions ** 2, dim=dims)
            t_sum = torch.sum(targets_onehot ** 2, dim=dims)
        else:
            i_sum = torch.sum(predictions, dim=dims)
            t_sum = torch.sum(targets_onehot, dim=dims)

        denom = i_sum + t_sum.float() + self.smooth

        # compute loss
        frac = nom / denom

        # apply weight for individual classesproperly
        if self.weight is not None:
            frac = self.weight * frac

        # average over classes
        frac = 1 - torch.mean(frac, dim=1)
        log.debug(f"{frac}")
        return torch.mean(frac)
