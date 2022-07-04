import os
from typing import List, Tuple
from torch.utils.data import Dataset
import torch

from datasets.data_models.raster_scenario import RasterScenarioData
from data_processing.online.rasterization import ScenarioRasterPreprocess
import configparser


class HeatmapOutputRasterScenarioDataset:
    def __init__(self, config: configparser.GlobalConfig):
        """
        Dataset for loading processed rasterized files

        Args:
            config: Config
        """
        self._preprocessor = ScenarioRasterPreprocess(config, disable_visualization=True)
        input_path = os.path.join(config.global_path, config.raster.data_process.input_path)
        self._scenario_paths = self._load_scenario_paths(input_path)

    def _load_scenario_paths(self, path) -> List[str]:
        return [os.path.join(path, filename) for filename in os.listdir(path)]

    def __getitem__(self, index: int) -> RasterScenarioData:
        raster_scenario_data, _ = self._preprocessor.process(self._scenario_paths[index])
        return raster_scenario_data

    def __len__(self) -> int:
        return len(self._scenario_paths)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class HeatmapOutputRasterScenarioDatasetTorchWrapper(Dataset):
    def __init__(self, config: configparser.GlobalConfig):
        """
        Dataset for loading processed files

        Args:
            config: Config
        """
        self.dataset = HeatmapOutputRasterScenarioDataset(config)

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        scenario_data = self.dataset[index]
        raster_features = torch.tensor(scenario_data.raster_features, dtype=torch.float32)
        da_area = torch.tensor(scenario_data.raster_features[:-1, ...], dtype=torch.float32)
        agent_traj_hist = torch.tensor(2 * scenario_data.agent_traj_hist / scenario_data.window_size, dtype=torch.float32)
        gt_heatmap = torch.unsqueeze(torch.tensor(scenario_data.heatmap, dtype=torch.float32), dim=0)

        return (raster_features, agent_traj_hist, da_area), gt_heatmap

    def __len__(self) -> int:
        return len(self.dataset)
