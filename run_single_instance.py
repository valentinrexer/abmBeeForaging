from bee_foraging_model.run import *
from bee_foraging_model.const import STEPS_PER_HOUR, STEPS_PER_DAY

def main():
    model = BeeForagingModel(source_distance=3333,
                             number_of_starting_bees=10,
                             flower_open= 7 * STEPS_PER_HOUR,
                             flower_closed= 9 * STEPS_PER_HOUR,
                             sucrose_concentration=0.25)

    model.run(5 * STEPS_PER_DAY + 1)

if __name__ == '__main__':
    main()