from dataclasses import dataclass
import yaml
import dacite
from typing import Optional
import numpy as np
import logging


@dataclass
class DataProcessParametersConfig:
    trajectory_history_window_length: int  # Nh
    trajectory_future_window_length: int  # Nr
    trajectory_min_history_window_length: int  # Nhmin
    object_trajectory_min_history_window_length: int  # Nhomin
    object_trajectory_min_future_window_length: int  # Nromin
    object_distance_threshold: Optional[float]  # Tsteps
    max_centerline_distance: float  # Dlsmax
    add_neighboring_lanes: bool


@dataclass
class DataProcessConfig:
    input_path: str
    output_path: str
    visualize: bool
    parameters: DataProcessParametersConfig


@dataclass
class LoggerConfig:
    level: str


@dataclass
class GlobalConfig:
    log: LoggerConfig
    data_process: DataProcessConfig

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

