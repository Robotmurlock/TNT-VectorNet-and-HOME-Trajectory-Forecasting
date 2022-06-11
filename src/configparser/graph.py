from dataclasses import dataclass


@dataclass
class DataProcessGraphConfig:
    input_path: str
    output_path: str
    max_polyline_segments: int
    max_polylines: int
    normalization_parameter: float
    visualize: bool


@dataclass
class GraphTrainConfigParameters:
    epochs: int
    n_targets: int
    anchor_generator_lr: float
    anchor_generator_sched_step: int
    anchor_generator_sched_gamma: float
    trajectory_forecaster_lr: float
    huber_delta: float


@dataclass
class GraphTrainConfig:
    input_path: str
    output_path: str
    parameters: GraphTrainConfigParameters
    visualize: bool


@dataclass
class GraphConfig:
    data_process: DataProcessGraphConfig
    train: GraphTrainConfig
