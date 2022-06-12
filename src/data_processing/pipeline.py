from abc import ABC, abstractmethod
from typing import Any, Iterable
from multiprocessing import Pool


class Pipeline(ABC):
    """
    Data processing pipeline abstraction
    """
    def __init__(self, output_path: str, visualize: bool = False):
        """

        Args:
            output_path: Path where output is stored
            visualize: Visualized processed data
        """
        self._output_path = output_path

        # visualization support
        self._fig_catalog = {}
        self._visualize = visualize

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
        if self._visualize:
            self.visualize(data)


def run_pipeline(pipeline: Pipeline, data_iterator: Iterable, n_processes: int) -> None:
    """
    Runs pipeline in multiple processes

    Args:
        pipeline: Pipeline
        data_iterator: DataIterator
        n_processes: Number of processes
    """
    with Pool(processes=n_processes) as pool:
        for _ in pool.imap(pipeline.process_and_save, data_iterator):
            pass
