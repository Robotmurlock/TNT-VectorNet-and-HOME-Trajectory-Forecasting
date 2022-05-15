from dataclasses import dataclass
from typing import List


@dataclass
class DataProcessRasterizationParametersConfig:
    agent_view_window_size: int
    object_shape: List[int]
    centerline_point_shape: List[int]
    gauss_kernel_size: int
    gauss_kernel_sigma: int


@dataclass
class DataProcessRasterizationConfig:
    input_path: str
    output_path: str
    visualize: bool
    debug_visualize: bool
    parameters: DataProcessRasterizationParametersConfig


@dataclass
class RasterizationTrainConfig:
    input_path: str
    output_path: str


@dataclass
class RasterConfig:
    data_process: DataProcessRasterizationConfig
    train: RasterizationTrainConfig
