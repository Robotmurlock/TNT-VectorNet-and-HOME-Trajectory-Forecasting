"""
Vectorized dataset
"""
import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from library.datasets.data_models.scenario import ScenarioData


class ScenarioDataset:
    def __init__(self, path: str):
        """
        Dataset for loading processed files

        Args:
            path: Path to dataset location on local file system
        """
        self.scenario_paths = self._index_scenario_paths(path)

    @staticmethod
    def _index_scenario_paths(path) -> List[str]:
        """
        Indexes dataset directory

        Args:
            path: Path to dataset

        Returns: List of scenario paths in dataset
        """
        return [os.path.join(path, filename) for filename in os.listdir(path)]

    def __getitem__(self, index: int) -> ScenarioData:
        return ScenarioData.load(self.scenario_paths[index])

    def __len__(self) -> int:
        return len(self.scenario_paths)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class ScenarioDatasetTorchWrapper(Dataset):
    def __init__(self, path: str):
        """
        Dataset for loading processed files

        Args:
            path: Path to dataset location on local file system
        """
        self.dataset = ScenarioDataset(path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scenario_data = self.dataset[index]
        agent_hist = torch.tensor(scenario_data.agent_traj_hist, dtype=torch.float32) / 25.0
        agent_future = scenario_data.ground_truth_trajectory_difference / 25.0
        agent_gt_end_point = scenario_data.final_point_gt.view(1, 2) / 25.0  # n_end_points, xy
        return agent_hist, agent_future, agent_gt_end_point

    def __len__(self) -> int:
        return len(self.dataset)
