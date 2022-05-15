from dataclasses import dataclass


@dataclass
class DataProcessGraphConfig:
    input_path: str
    output_path: str
    visualize: bool


@dataclass
class GraphTrainConfig:
    input_path: str
    output_path: str


@dataclass
class GraphConfig:
    data_process: DataProcessGraphConfig
    train: GraphTrainConfig
