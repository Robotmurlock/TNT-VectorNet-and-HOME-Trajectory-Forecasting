"""
Torch Dataset (dataloader) for VectorNet training
"""
import os
from typing import Tuple

import torch

from datasets.data_models.graph_scenario import GraphScenarioData
from torch.utils.data import Dataset


class GraphScenarioDataset:
    def __init__(self, path: str):
        """
        Dataset independent of any framework
        """
        self.scenario_paths = [os.path.join(path, filename) for filename in os.listdir(path)]

    def __len__(self):
        return len(self.scenario_paths)

    def __getitem__(self, index: int) -> GraphScenarioData:
        """
        VectorNetScenarioDataset allows using raw GraphScenarioData values

        Args:
            index: Scenario Index

        Returns: Loads scenario from given path index
        """
        return GraphScenarioData.load(self.scenario_paths[index])

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class VectorNetScenarioDataset(Dataset):
    def __init__(self, path: str):
        """
        Dataset for loading processed graph polylines

        Args:
            path: Path to dataset location on local file system
        """
        self._dataset = GraphScenarioDataset(path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self._dataset[index]
        polylines, anchors, ground_truth, gt_traj = \
            data.inputs, data.target_proposals, \
            data.target_ground_truth, data.ground_truth_trajectory_difference
        return polylines, anchors, ground_truth, gt_traj

    def __len__(self) -> int:
        return len(self._dataset)
