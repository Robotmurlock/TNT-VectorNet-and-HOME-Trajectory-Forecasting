from dataclasses import dataclass


@dataclass
class DataProcessGraphConfig:
    input_path: str
    output_path: str
    normalization_parameter: float
    visualize: bool


@dataclass
class GraphTrainConfigParameters:
    epochs: int
    n_targets: int
    anchor_generator_lr: float
    trajectory_forecaster_lr: float


@dataclass
class GraphTrainConfig:
    input_path: str
    output_path: str
    parameters: GraphTrainConfigParameters


@dataclass
class GraphConfig:
    data_process: DataProcessGraphConfig
    train: GraphTrainConfig
