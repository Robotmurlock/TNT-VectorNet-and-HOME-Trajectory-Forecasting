import torch.nn as nn
import torch
from typing import Tuple


class BaseModel(nn.Module):
    def __init__(self, trajectory_length: int):
        """
        # FIXME: Deprecated

        Args:
            trajectory_length: Forecasted trajectory length
        """
        super(BaseModel, self).__init__()
        self._prediction_length = trajectory_length

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('Base model does not have implemented forward() method!')

    def load_state(self, path: str) -> None:
        raise NotImplementedError('Base model does not have implemented load_state() method!')
