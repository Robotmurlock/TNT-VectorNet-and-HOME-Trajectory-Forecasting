"""
VectorNet config dataclass
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataProcessGraphConfig:
    input_path: str
    output_path: str
    max_polyline_segments: int
    max_polylines: int
    normalization_parameter: float
    sampling_algorithm: str
    visualize: bool
    report: bool

    skip: Optional[List[str]] = field(default=None)
    visualize_anchors: bool = field(default=False)
    visualize_candidate_centerlines: bool = field(default=False)


@dataclass
class GraphTrainConfigParameters:
    epochs: int
    batch_size: int
    n_targets: int
    n_trajectories: int
    tg_lr: float
    tg_sched_step: int
    tg_sched_gamma: float
    tf_lr: float
    tf_sched_step: int
    tf_sched_gamma: float
    tfs_lr: float
    tfs_sched_step: int
    tfs_sched_gamma: float
    huber_delta: float
    use_traj_scoring: bool


@dataclass
class GraphTrainConfig:
    train_input_path: str
    val_input_path: str
    model_name: str
    parameters: GraphTrainConfigParameters
    visualize: bool
    n_workers: int

    resume: bool = field(default=True)
    starting_checkpoint_name: str = field(default='last.ckpt')


@dataclass
class GraphConfig:
    data_process: DataProcessGraphConfig
    train: GraphTrainConfig
