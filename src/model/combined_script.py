import sys
import time
import random

import mesa
from mesa.time import SimultaneousActivation
from mesa import DataCollector, Agent
import multiprocessing as mp



# Agent classes
# ___________________________________________________________________________________________________


#Honey bee agent
# ___________________________________________________________________________________________________

class HoneyBeeAgent(mesa.Agent):
    def __init__(self, bee_agent_id, bee_model, max_nectar):
        super().__init__(bee_agent_id, bee_model)
        self.status = 's'
        self.nectar = 0
        self.max_nectar = max_nectar
        self.flower = None

    # Search for flowers in the current cell or move randomly
    def search(self):

        # if bee has no flower, yet, it just randomly searches for one
        if self.flower is None:

            # check if flower is available at grid cell
            agents_at_pos = self.model.grid.get_cell_list_contents(self.pos)
            available_flowers = [agent for agent in agents_at_pos if isinstance(agent, FlowerAgent)]
            if available_flowers:
                self.flower = random.choice(available_flowers)

            else:
                # Move randomly if no flower is found
                new_pos = (self.pos[0] + random.randint(-1, 1), self.pos[1] + random.randint(-1, 1))
                if not self.model.grid.out_of_bounds(new_pos):
                    self.model.grid.move_agent(self, new_pos)  # Correct agent movement

        else:
            if self.pos == self.flower.pos:
                if self.flower.nectar > 0:
                    self.nectar += 1
                    self.flower.nectar -= 1

            else:
                x_inc = 0
                y_inc = 0

                if self.pos[0] < self.flower.pos[0]:
                    x_inc = 1

                elif self.pos[0] > self.flower.pos[0]:
                    x_inc = -1

                if self.pos[1] < self.flower.pos[1]:
                    y_inc = 1

                elif self.pos[1] > self.flower.pos[1]:
                    y_inc = -1

                self.model.grid.move_agent(self, (self.pos[0] + x_inc, self.pos[1] + y_inc))



    def return_to_hive(self):
        x_move = 0
        y_move = 0

        # Navigate back to the hive
        if self.pos != self.model.hive:
            if self.pos[0] < self.model.hive[0]:
                x_move = 1
            elif self.pos[0] > self.model.hive[0]:
                x_move = -1

            if self.pos[1] < self.model.hive[1]:
                y_move = 1
            elif self.pos[1] > self.model.hive[1]:
                y_move = -1

        new_pos = (self.pos[0] + x_move, self.pos[1] + y_move)
        self.model.grid.move_agent(self, new_pos)  # Move agent to new position

    def unload(self):
        if self.nectar > 0:
            self.nectar -= 1
            self.model.hive_stock += 1  # Update the hive stock in the model

    def update_status(self):
        # Update the agent's status based on nectar and position
        if self.nectar == self.max_nectar:
            if self.pos == self.model.hive:
                self.status = 'u'  # Unloading at the hive
            else:
                self.status = 'c'  # Carrying nectar to the hive
        elif self.nectar > 0 and self.pos == self.model.hive:
            self.status = 'u'
        else:
            self.status = 's'  # Searching for nectar

    def step(self):
        self.update_status()

        if self.status == 'u':
            self.unload()

        if self.status == 'c':
            self.return_to_hive()

        if self.status == 's':
            self.search()


# Flower agent
# ______________________________________________________________________________________________________
class FlowerAgent(mesa.Agent):
    def __init__(self, flower_agent_id, bee_model):
        super().__init__(flower_agent_id, bee_model)
        self.nectar = random.randint(0,100)


    def step(self):
        self.nectar += random.randint(0,2)







# Model class

class BeeModel(mesa.Model):
    def __init__(self, height, width, n_flowers, n_bees):
        super().__init__()
        self.hive = (round(width / 2), round(height / 2))
        self.hive_stock = 0

        # model activation type ==> Simultaneous means every agent performs its action at the same time i.e. there is no order
        self.schedule = SimultaneousActivation(self)

        # model grid
        self.height = height
        self.width = width
        self.grid = mesa.space.MultiGrid(height=self.height, width=self.width, torus=False)

        # data collection of data of interest
        self.datacollector = DataCollector(
            model_reporters={
                "HiveStock: ": self.get_hive_stock,
                "Bees status: ": self.get_bees_status,
                "Bees nectar: ": self.get_bees_nectar,
            }
        )
        for i in range(n_flowers):
            flower_agent = FlowerAgent(i, self)
            self.schedule.add(flower_agent)

            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.grid.place_agent(flower_agent, (x, y))

        for i in range(n_bees):
            honey_bee_agent = HoneyBeeAgent(i, self, random.randint(4, 8))
            self.schedule.add(honey_bee_agent)

            self.grid.place_agent(honey_bee_agent, self.hive)

    def print_grid(self):
        for y in range(self.grid.height):
            row = ""
            for x in range(self.grid.width):
                cell_contents = self.grid.get_cell_list_contents([(x, y)])
                # Check if any of the agents in the cell is a FlowerAgent
                if (x, y) == self.hive:
                    row += "H "

                elif any(isinstance(agent, FlowerAgent) for agent in cell_contents):
                    row += "X "

                elif any(isinstance(agent, HoneyBeeAgent) for agent in cell_contents):
                    row += "_ "

                else:
                    row += "0 "
            print(row)

    def get_hive_stock(self):
        return self.hive_stock

    def get_bees_status(self):
        return [bee.status for bee in self.schedule.agents if isinstance(bee, HoneyBeeAgent)]

    def get_bees_nectar(self):
        return [bee.nectar for bee in self.schedule.agents if isinstance(bee, HoneyBeeAgent)]

    def get_data_from_collector(self):
        return self.datacollector.get_model_vars_dataframe()

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)





# Runner methods
# _________________________________________________________________________________________________

def run_model(params):
    model = BeeModel(params['height'], params['width'], params['n_flowers'], params['n_bees'])
    for _ in range(params['steps']):
        model.step()

    return model.get_data_from_collector()


def parallel_run(num_processes, num_iterations, model_params):
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_model, [model_params] * num_iterations)
    return results



# Main
# ________________________________________________________________________________________________

def __main__(args):
    start_time = time.time()

    if len(args) < 2:
        print("Usage: python script.py <steps> [num_iterations]")
        return

    try:
        k = int(args[1])
        num_iterations = int(args[2]) if len(args) > 2 else 5

    except ValueError:
        print("Invalid arguments. Please provide integers for steps and iterations.")
        return

    num_processes = mp.cpu_count()  # Use all available CPU cores

    model_params = {
        'height': 2000,
        'width': 2000,
        'n_flowers': 7500,
        'n_bees': 2500,
        'steps': k
    }

    results = parallel_run(num_processes, num_iterations, model_params)


    # Print results
    for i, data_frame in enumerate(results):
        print(f"Run: {i}")
        print(data_frame)


    print("")
    time_elapsed = time.time() - start_time

    hours, remainder = divmod(time_elapsed, 3600)  # Get hours and remainder in seconds
    minutes, seconds = divmod(remainder, 60)  # Get minutes and remainder in seconds

    print(f"Execution took {int(hours)} h {int(minutes)} m {int(seconds)} s")


if __name__ == '__main__':
    mp.freeze_support()  # Add this line
    __main__(sys.argv)