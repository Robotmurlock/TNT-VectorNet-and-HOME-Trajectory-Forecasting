import torch.nn as nn
import torch
from typing import Tuple


class HomeExpertSystem(nn.Module):
    def __init__(self, n_trajectories: int, area_size: int, non_da_penalty: float = 0.04):
        super().__init__()

        self._n_trajectories = n_trajectories
        self._area_size = area_size
        self._area_halfsize = area_size // 2

        # hyperparameters
        self._non_da_penalty = non_da_penalty

    def forward(self, trajectories: torch.Tensor, confidences: torch.Tensor, targets: torch.Tensor, da_area: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_trajectories, traj_length = trajectories.shape[0], trajectories.shape[-2]
        assert n_trajectories >= self._n_trajectories, f'HOME Expert system expects at least {self._n_trajectories} but found {n_trajectories}!'

        for traj_index in range(n_trajectories):
            for point_index in range(traj_length):
                coords = [int(c) + self._area_halfsize for c in trajectories[traj_index, point_index].detach().cpu().tolist()]
                row, col = coords[::-1]  # swap row/col
                if da_area[row, col] == 0.0:
                    confidences[traj_index] -= self._non_da_penalty

        indices = torch.argsort(confidences, descending=True)
        trajectories = trajectories[indices][:self._n_trajectories]
        confidences = confidences[indices][:self._n_trajectories]
        targets = targets[:self._n_trajectories]

        return trajectories, confidences, targets
