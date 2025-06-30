from typing import Any
import os
import csv
import logging
import multiprocessing as mp

from .bee_foraging_model import BeeForagingModel

mp.log_to_stderr(logging.INFO)
_mp_LOGGER = mp.get_logger()

class DataCollector:
    """
    Collects data from a BeeForagingModel instance in specified intervals
    """
    def __init__(self, model : BeeForagingModel, path_to_csv :str, collection_interval : int) -> None:
        if not isinstance(model, BeeForagingModel):
            raise TypeError("BeeForagingModel must be a BeeForagingModel")

        if not os.path.exists(path_to_csv):
            raise AttributeError(f"File {path_to_csv} does not exist")

        self.model = model
        self.path_to_csv = path_to_csv
        self.collection_interval = collection_interval
        self.columns = ['number_of_starting_foragers', 'source_distance', 'sucrose_concentration', 'anticipation_method', 'flower_open', 'flower_open' , 'time_step', 'energy']


        file_is_empty = not os.path.exists(path_to_csv) or os.path.getsize(path_to_csv) == 0
        if file_is_empty:
            self.make_header()

    def make_header(self) -> None:
        """
        Creates a header for the csv file
        """
        with open (self.path_to_csv, "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self.columns)

    def collect_data(self) -> None:
        """
        Collects data of all columns specified in self.columns
        """
        row = [self.model.number_of_starting_bees,
               self.model.initial_source_distance,
               self.model.sucrose_concentration,
               self.model.anticipation_method,
               self.model.flowers[0].open_time,
               self.model.flowers[0].close_time,
               self.model.steps,
               self.model.total_energy]

        with open (self.path_to_csv, "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)

    def check_for_collection_call(self) -> None:
        """
        Checks if the model has reached a collection time
        """
        if self.model.steps == 0:
            return

        if self.model.steps % self.collection_interval == 0:
            self.collect_data()

def run_model_instance(time_steps : int, **params) -> float:
    """

    :param time_steps:
    :param params:
    :return:
    """
    model = BeeForagingModel(**params)
    model.run(time_steps)
    return model.total_energy

def run_single_model_instance(args: tuple[int, dict[str, int | str | float | Any]]) -> float:
    """
    Wrapper function that unpacks arguments for run_model_instance

    :param args: A tuple containing (time_steps, specific_params)
    :return: Result of the model run
    """

    time_steps, specific_params = args
    _mp_LOGGER.info(f"Running model instance for parameters: {specific_params}")
    return run_model_instance(time_steps, **specific_params)

def parallel_run(num_cores: int,
                 num_runs_per_combination: int,
                 time_steps: int,
                 params: list[dict[str, int | str | float | Any]]) -> list[float]:
    """
    Run parallel simulations

    :param num_cores: Number of CPU cores to use
    :param num_runs_per_combination: Number of times to run each parameter combination
    :param time_steps: Number of simulation steps
    :param params: List of parameter dictionaries
    :return: Aggregated results from all runs
    """
    # Expand params to include multiple runs
    expanded_params = [(time_steps, param) for param in params for _ in range(num_runs_per_combination)]

    # Create a process pool
    with mp.Pool(processes=num_cores) as pool:
        # Use map with the wrapper function
        results = pool.map(run_single_model_instance, expanded_params)

    return results