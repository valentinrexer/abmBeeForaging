from bee_foraging_model.run import *
import sys
from itertools import product
from bee_foraging_model.const import STEPS_PER_HOUR, STEPS_PER_DAY
import logging
from datetime import datetime
import argparse
import json

_MAIN_LOGGER = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Parser for csv-file and JSON with foraging model parameters')
    parser.add_argument('--csv', type=str)
    parser.add_argument('--json', type=str)

    args = parser.parse_args()

    params: dict
    with open(args.json) as json_file:
        params = json.load(json_file)

    number_of_starting_foragers = params['number_of_starting_foragers']
    source_distance = params['source_distance']
    sucrose_concentration = params['sucrose_concentration']
    anticipation_method = params['anticipation_method']
    anthesis_interval = params['anthesis_interval']

    for i in range(len(anthesis_interval)):
        curr_interval = anthesis_interval[i]
        anthesis_interval[i] = (curr_interval[0] * STEPS_PER_HOUR,
                                curr_interval[1] * STEPS_PER_HOUR)

    number_of_steps = 5 * STEPS_PER_DAY + 1
    number_of_runs_per_combination = 10

    params = []
    for n, d, c, a, a_in in product(
        number_of_starting_foragers,
        source_distance,
        sucrose_concentration,
        anticipation_method,
        anthesis_interval
    ):
        params.append({
            "number_of_starting_bees": n,
            "source_distance": d,
            "sucrose_concentration": c,
            "anticipation_method": a,
            "flower_open": a_in[0],
            "flower_closed": a_in[1],
            "collector_path" : args.csv,
            "collection_interval": STEPS_PER_DAY
        })

    parallel_run(
        mp.cpu_count() - 2,
        number_of_runs_per_combination,
        number_of_steps,
        params
    )

    _MAIN_LOGGER.critical(f"Finished all simulations! Timestamp: {datetime.now()}")

if __name__ == "__main__":
    main()