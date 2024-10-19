import mesa
import random

class HoneyBeeAgent(mesa.Agent):

    def __init__(self, bee_agent_id, model):
        super().__init__(bee_agent_id, model)

    #todo: search (fly away from the center) in any direction

    #todo: fly back in the fastest way possible

    #todo: rest and let nectar sit at hive

    #todo: load itself with nectar



class FlowerAgent(mesa.Agent):
    def __init__(self, flower_agent_id, model):
        super().__init__(flower_agent_id, model)
        self.nectar = random.randint(0,100)


    def step(self):
        self.nectar += random.randint(0,2)


