"""
Heatmap output config
"""
from dataclasses import dataclass
from typing import List


@dataclass
class DataProcessRasterizationParametersConfig:
    agent_view_window_size: int
    object_shape: List[int]
    centerline_point_shape: List[int]
    gauss_kernel_size: int
    gauss_kernel_sigma: int
    max_neighbours: int

    @property
    def agent_view_window_halfize(self) -> int:
        return self.agent_view_window_size // 2


@dataclass
class DataProcessRasterizationConfig:
    input_path: str
    output_path: str
    visualize: bool
    debug_visualize: bool
    parameters: DataProcessRasterizationParametersConfig


@dataclass
class RasterizationTrainHeatmapConfig:
    batch_size: int
    n_workers: int
    epochs: int
    augmentation: bool
    input_path: str
    output_path: str


@dataclass
class RasterizationTrainTrajectoryForecasterParametersConfig:
    epochs: int
    batch_size: int
    lr: float
    sched_step: int
    sched_gamma: float


@dataclass
class RasterizationTrainTrajectoryForecasterConfig:
    train_input_path: str
    val_input_path: str
    output_path: str
    n_workers: int
    model_name: str
    parameters: RasterizationTrainTrajectoryForecasterParametersConfig


@dataclass
class RasterConfig:
    data_process: DataProcessRasterizationConfig
    train_heatmap: RasterizationTrainHeatmapConfig
    train_tf: RasterizationTrainTrajectoryForecasterConfig
