import os
from typing import List, Tuple, Union

import torch

from datasets.data_models.graph_scenario import GraphScenarioData
from torch.utils.data import Dataset


class VectorNetScenarioDataset(Dataset):
    def __init__(self, path: str, device: Union[str, torch.device]):
        """
        Dataset for loading processed rasterized files

        Args:
            path: Path to dataset location on local file system
        """
        self.scenario_paths = self._load_scenario_paths(path)
        self._device = device

    def _load_scenario_paths(self, path) -> List[str]:
        return [os.path.join(path, filename) for filename in os.listdir(path)]

    def scenario(self, index: int) -> GraphScenarioData:
        return GraphScenarioData.load(self.scenario_paths[index])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.scenario(index)
        polylines, anchors, ground_truth, gt_traj = \
            data.inputs.to(self._device), data.target_proposals.to(self._device), \
            data.target_ground_truth.to(self._device), data.ground_truth_trajectory_difference.to(self._device)
        return polylines, anchors, ground_truth, gt_traj

    def __len__(self) -> int:
        return len(self.scenario_paths)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
