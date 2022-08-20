from dataclasses import dataclass, field
import yaml
import dacite
from typing import Optional, List
import numpy as np
import logging

import utils.steps
from configparser.raster import RasterConfig
from configparser.graph import GraphConfig


@dataclass
class LoggerConfig:
    level: str


@dataclass
class GlobalParametersConfig:
    trajectory_history_window_length: int  # Nh
    trajectory_future_window_length: int  # Nr


@dataclass
class DataProcessParametersConfig:
    trajectory_min_history_window_length: int  # Nhmin
    object_trajectory_min_history_window_length: int  # Nhomin
    object_trajectory_min_future_window_length: int  # Nromin
    object_distance_threshold: Optional[float]  # Tsteps
    centerline_radius_scale: float  # Dlsmax
    add_neighboring_lanes: bool
    lane_radius_scale: float
    min_lane_radius: float


@dataclass
class DataProcessConfig:
    input_path: str
    output_path: str
    n_processes: int
    visualize: bool
    parameters: DataProcessParametersConfig
    skip: Optional[List[str]]


@dataclass
class EvaluationConfig:
    input_path: str
    output_path: str
    visualize: bool


@dataclass
class GlobalConfig:
    log: LoggerConfig
    global_parameters: GlobalParametersConfig
    data_process: DataProcessConfig
    evaluation: EvaluationConfig
    raster: Optional[RasterConfig]
    graph: Optional[GraphConfig]

    global_path: Optional[str] = field(default=utils.steps.SOURCE_PATH)
    model_storage_path: Optional[str] = field(default=utils.steps.MODEL_PATH)

    def __post_init__(self):
        self.configure_logging()

    def configure_logging(self) -> None:
        """
        Configures logger
        """
        np.set_printoptions(formatter={'float': '{:0.3f}'.format})
        logging.basicConfig(
            level=logging.getLevelName(self.log.level),
            format='%(asctime)s [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.getLogger('matplotlib').setLevel(logging.CRITICAL)


def config_from_yaml(path: str) -> GlobalConfig:
    """
    Creates global config object from yaml file path

    Args:
        path: Yaml file location

    Returns: Global configs
    """
    with open(path, 'r', encoding='utf-8') as file_stream:
        raw = yaml.safe_load(file_stream)
    return dacite.from_dict(data_class=GlobalConfig, data=raw)
