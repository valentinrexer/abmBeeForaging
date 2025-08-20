"""
    contains methods to run simulations with different parameters.
    Implements support for processing on multiple cores.
"""

from typing import Any
import logging
import multiprocessing as mp

from bee_foraging_model.bee_foraging_model import BeeForagingModel

mp.log_to_stderr(logging.INFO)
_mp_LOGGER = mp.get_logger()

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