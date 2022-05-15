import os
from typing import List

from datasets.data_models.graph_scenario import GraphScenarioData


class VectorNetScenarioDataset:
    def __init__(self, path: str):
        """
        Dataset for loading processed rasterized files

        Args:
            path: Path to dataset location on local file system
        """
        self.scenario_paths = self._load_scenario_paths(path)

    def _load_scenario_paths(self, path) -> List[str]:
        return [os.path.join(path, filename) for filename in os.listdir(path)]

    def __getitem__(self, index: int) -> GraphScenarioData:
        return GraphScenarioData.load(self.scenario_paths[index])

    def __len__(self) -> int:
        return len(self.scenario_paths)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
