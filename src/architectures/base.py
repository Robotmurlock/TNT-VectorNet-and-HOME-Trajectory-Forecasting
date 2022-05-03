import torch.nn as nn
import torch
from typing import Tuple


class BaseModel(nn.Module):
    def __init__(self, prediction_length: int):
        """
        Args:
            prediction_length: Forecasted trajectory length
        """
        super(BaseModel, self).__init__()
        self._prediction_length = prediction_length

    def forecast(self, features: Tuple[torch.Tensor, ...], is_batch: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forecasts results for scenario

        Args:
            features: All features (agent and objects)
            is_batch: Is input wrapped in batch or not

        Returns: Scenario forecasts
        """
        raise NotImplementedError('Base model does not have implemented forecast() method!')
