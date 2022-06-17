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
    tg_lr: float
    tg_sched_step: int
    tg_sched_gamma: float
    tf_lr: float
    tf_sched_step: int
    tf_sched_gamma: float
    huber_delta: float


@dataclass
class GraphTrainConfig:
    train_input_path: str
    val_input_path: str
    output_path: str
    parameters: GraphTrainConfigParameters
    visualize: bool


@dataclass
class GraphConfig:
    data_process: DataProcessGraphConfig
    train: GraphTrainConfig
