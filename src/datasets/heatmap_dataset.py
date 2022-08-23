import os
from typing import List, Dict
from torch.utils.data import Dataset
import torch
import random

from datasets.data_models.raster_scenario import RasterScenarioData
from data_processing.online.heatmap_rasterization import ScenarioRasterPreprocess
import configparser


class HeatmapOutputRasterScenarioDataset:
    def __init__(self, config: configparser.GlobalConfig, split: str):
        """
        Dataset for loading processed rasterized files

        Args:
            config: Config
        """
        self._preprocessor = ScenarioRasterPreprocess(config, disable_visualization=True)
        input_path = os.path.join(config.global_path, config.raster.data_process.input_path, split)
        self._scenario_paths = self._load_scenario_paths(input_path)

    def _load_scenario_paths(self, path) -> List[str]:
        return [os.path.join(path, filename) for filename in os.listdir(path)]

    def __getitem__(self, index: int) -> RasterScenarioData:
        raster_scenario_data = self._preprocessor.process(self._scenario_paths[index])
        return raster_scenario_data

    def __len__(self) -> int:
        return len(self._scenario_paths)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

class HeatmapOutputRasterScenarioDatasetTorchWrapper(Dataset):
    def __init__(self, config: configparser.GlobalConfig, split: str):
        """
        Dataset for loading processed files

        Args:
            config: Config
        """
        self.dataset = HeatmapOutputRasterScenarioDataset(config, split)
        self._augmentation = config.raster.train_heatmap.parameters.augmentation and split=='train'

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        scenario_data = self.dataset[index]
        if self._augmentation:
            r = random.uniform(0, 1)
            if r < 5:
                scenario_data.flip()

        raster_features = torch.tensor(scenario_data.raster_features, dtype=torch.float32)
        da_area = torch.tensor(scenario_data.raster_features[0, ...], dtype=torch.float32)
        agent_traj_hist = torch.tensor(scenario_data.agent_traj_hist / 25.0, dtype=torch.float32)
        objects_traj_hist = torch.tensor(scenario_data.objects_traj_hists / 25.0, dtype=torch.float32)
        gt_heatmap = torch.unsqueeze(torch.tensor(scenario_data.heatmap, dtype=torch.float32), dim=0)
        agent_traj_gt_end_point = torch.tensor(scenario_data.agent_traj_gt[-1, :], dtype=torch.float32)
        agent_traj_gt = torch.tensor(scenario_data.agent_traj_gt, dtype=torch.float32)

        return {
            'raster': raster_features,
            'agent_traj_hist': agent_traj_hist,
            'agent_traj_diff': scenario_data.ground_truth_trajectory_difference,
            'agent_traj_gt_end_point': agent_traj_gt_end_point,
            'objects_traj_hist': objects_traj_hist,
            'agent_traj_gt': agent_traj_gt,
            'da_area': da_area,
            'heatmap': gt_heatmap,
            'id': scenario_data.id,
            'city': scenario_data.city,
            'heatmap_gt': torch.tensor(scenario_data.heatmap, dtype=torch.float32)
        }

    def __len__(self) -> int:
        return len(self.dataset)
