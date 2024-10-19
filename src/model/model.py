import mesa
import random

from mesa.time import SimultaneousActivation
from src.agents.agent import FlowerAgent, HoneyBeeAgent


class BeeModel(mesa.Model):
    def __init__(self, height, width, n_flowers, n_bees):
        super().__init__()
        self.height = height
        self.width = width
        self.hive = (round(width/2), round(height/2))
        self.hive_stock = 0

        self.schedule = SimultaneousActivation(self)

        self.grid = mesa.space.MultiGrid(height=self.height, width=self.width, torus=False)

        for i in range(n_flowers):
            flower_agent = FlowerAgent(i, self)
            self.schedule.add(flower_agent)

            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            self.grid.place_agent(flower_agent, (x, y))

        for i in range(n_bees):
            honey_bee_agent = HoneyBeeAgent(i, self, random.randint(4,8))
            self.schedule.add(honey_bee_agent)

            self.grid.place_agent(honey_bee_agent, self.hive)


    def print_grid(self):
        for y in range(self.grid.height):
            row = ""
            for x in range(self.grid.width):
                cell_contents = self.grid.get_cell_list_contents([(x, y)])
                # Check if any of the agents in the cell is a FlowerAgent
                if (x,y) == self.hive:
                    row += "H "

                elif any(isinstance(agent, FlowerAgent) for agent in cell_contents):
                    row += "X "

                elif any(isinstance(agent, HoneyBeeAgent) for agent in cell_contents):
                    row += "_ "
                    
                else:
                    row += "0 "
            print(row)




values = [50, 200, 500, 2000, 5000, 50000]
for f in range(10):
    print(f)
    print("")
    for k in values:
        model = BeeModel(2000, 2000, 7500, 2500)
        for i in range(k):
            model.schedule.step()

        print(f'{k}: {model.hive_stock}')

    print("")
