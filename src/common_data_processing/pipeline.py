"""
Simple abstraction for multiprocessing pipelines
"""
import logging
from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import Any, Collection

from tqdm import tqdm

logger = logging.getLogger('PipelineModule')


class Pipeline(ABC):
    """
    Data processing pipeline abstraction
    """
    def __init__(self, output_path: str, visualize: bool = False, report: bool = False):
        """

        Args:
            output_path: Path where output is stored
            visualize: Visualized processed data
        """
        self._output_path = output_path

        # visualization support
        self._fig = None
        self.viz = visualize

        self._report = report

    @property
    def has_report(self) -> bool:
        """
        Returns: report status
        """
        return self._report

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Processes loaded data

        Args:
            data: Data

        Returns: Processed data
        """
        pass

    @abstractmethod
    def save(self, data: Any) -> None:
        """
        Saves processed data

        Args:
            data: Data to be saved
        """
        pass

    def report(self) -> None:
        """
        Shows pipeline report (optional)

        Returns:
        """
        raise NotImplemented('Report is not implemented')

    def visualize(self, data: Any) -> None:
        """
        Optional: Data visualization
        Args:
            data: Data
        """
        raise NotImplemented('Visualization is not implemented')

    def process_and_save(self, data: Any) -> None:
        """
        Process -> Save -> Visualize (optional)

        Args:
            data: Data
        """
        data = self.process(data)
        if data is None:
            return

        self.save(data)
        if self.viz:
            self.visualize(data)


def run_pipeline(pipeline: Pipeline, data_iterator: Collection, n_processes: int) -> None:
    """
    Runs pipeline in multiple processes

    Args:
        pipeline: Pipeline
        data_iterator: DataIterator
        n_processes: Number of processes
    """
    if n_processes == 1:
        logger.warning('Process is run sequentially because only one process is being used!')
        for data in data_iterator:
            pipeline.process_and_save(data)
    else:
        assert not pipeline.viz, f'Data visualization is supported for only one process. Got {n_processes}.'

        with Pool(processes=n_processes) as pool:
            for _ in tqdm(pool.imap(pipeline.process_and_save, data_iterator), total=len(data_iterator)):
                pass

    if pipeline.has_report:
        pipeline.report()
