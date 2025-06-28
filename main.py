from bee_foraging_model.bee_foraging_model import *
import itertools

def main(args):
    csv_path = args[1]

    # Number of starting foragers
    number_of_starting_foragers = [10, 33, 100, 333, 1000]

    # Source distance from the hive
    source_distance = [33, 100, 333, 1000, 3333]

    # sucrose concentration
    sucrose_concentration = [0.25, 0.5, 1.0, 2.0]

    # anticipation strategy
    anticipation_method = [1, 2]

    # anthesis time interval
    anthesis_interval = [(7, 9), (7, 11), (7, 15),
                         (12, 14), (11, 15), (9, 17),
                         (17, 19), (15, 19), (11, 19)]

    number_of_steps = 5 * 1 # will be replaced by 5 * STEPS_PER_DAY
    number_of_runs_per_combination = 10

    params = []
    for n, d, c, a, a_in in itertools.product(
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
            "collector_path" : csv_path,
            "collection_interval": STEPS_PER_DAY
        })

    parallel_run(
        mp.cpu_count() - 2,
        number_of_runs_per_combination,
        number_of_steps,
        params
    )

if __name__ == "__main__":
    main(sys.argv)