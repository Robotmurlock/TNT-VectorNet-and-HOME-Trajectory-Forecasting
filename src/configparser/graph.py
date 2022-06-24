"""
VectorNet config dataclass
"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DataProcessGraphConfig:
    input_path: str
    output_path: str
    max_polyline_segments: int
    max_polylines: int
    normalization_parameter: float
    visualize: bool
    visualize_anchors: bool
    skip: Optional[List[str]]


@dataclass
class GraphTrainConfigParameters:
    epochs: int
    batch_size: int
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
    n_workers: int


@dataclass
class GraphConfig:
    data_process: DataProcessGraphConfig
    train: GraphTrainConfig
