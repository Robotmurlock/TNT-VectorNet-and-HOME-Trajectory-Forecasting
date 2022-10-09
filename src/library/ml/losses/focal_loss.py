import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float, reduction: str = 'mean'):
        super(BinaryFocalLoss, self).__init__()
        self._alpha = alpha

        valid_reduction_choices = ['mean', 'sum', 'none']
        if reduction not in valid_reduction_choices:
            raise ValueError(f'Invalid reduction choice: "{reduction}". Valid choices: {valid_reduction_choices}.')
        self._reduction = reduction

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, min=1e-3, max=1 - 1e-3)

        # noinspection PyUnresolvedReferences
        mask = (labels == 1.0).float()
        positive_label_loss = - mask * torch.log(pred)
        negative_label_loss = - (1-mask) * torch.pow(pred, self._alpha) * torch.log(1 - pred)
        loss = positive_label_loss + negative_label_loss

        if self._reduction == 'none':
            return loss
        elif self._reduction == 'sum':
            return torch.sum(loss)
        elif self._reduction == 'mean':
            return torch.mean(loss)
        else:
            raise AssertionError('Invalid Program State!')