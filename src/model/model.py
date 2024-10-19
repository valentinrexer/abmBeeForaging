import mesa
import random

from mesa.time import SimultaneousActivation

from src.agents.agent import FlowerAgent, HoneyBeeAgent


class BeeModel(mesa.Model):
    def __init__(self, height, width, n_flowers):
        super().__init__()
        self.height = height
        self.width = width

        self.schedule = SimultaneousActivation(self)

        self.grid = mesa.space.MultiGrid(height=self.height, width=self.width, torus=False)

        for i in range(n_flowers):
            flower_agent = FlowerAgent(i, self)
            self.schedule.add(flower_agent)

            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            self.grid.place_agent(flower_agent, (x, y))
            self.grid.place_agent(HoneyBeeAgent(100, self), (0,0))

    def print_grid(self):
        for y in range(self.grid.height):
            row = ""
            for x in range(self.grid.width):
                cell_contents = self.grid.get_cell_list_contents([(x, y)])
                # Check if any of the agents in the cell is a FlowerAgent
                if any(isinstance(agent, FlowerAgent) for agent in cell_contents):
                    row += "X "

                elif any(isinstance(agent, HoneyBeeAgent) for agent in cell_contents):
                    row += "H "
                    
                else:
                    row += "0 "
            print(row)


model = BeeModel(20, 20, 50)

model.print_grid()