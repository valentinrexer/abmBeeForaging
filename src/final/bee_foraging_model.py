#packages for modeling
import mesa

#packages for feature implementation
from enum import Enum
import random

#packages for data analysis
import pandas as pd
import numpy as np




### definition of global attributes ###

#flower colors
class Color(Enum):
    RED = 1,
    GREEN = 2,
    BLUE = 3,
    PURPLE = 4,
    YELLOW = 5,
    WHITE = 6,


#flower closed
class Bloom(Enum):
    OPEN = True,
    CLOSED = False



#bee type
class BeeType(Enum):
    EXPERIENCED_FORAGER = 1,
    NEW_FORAGER = 2,


#bee status
class BeeStatus(Enum):
    RESTING = 1,
    CLUSTERING = 2,
    DANCING = 3,
    FLYING_OUT = 4,
    RETURNING = 5,
    LOADING_NECTAR = 6,
    UNLOADING_NECTAR = 7,
    SEARCHING_ADVERTISED_SOURCE = 8,




# definition of the Flower Class
#todo: adjust flower stock variable
"""
    Represents a single flower in the model 
    
    Args:
        flower_id (int): unique id of the flower agent
        bee_model (mesa.Model): model where the agent is placed
        sucrose_concentration (float): sucrose concentration in the flower
        
        closed (bool) [optional]: flower is closed
        color (Color) [optional]: color of the flower
"""

class FlowerAgent(mesa.Agent):
    def __init__(self, flower_id, bee_model, sucrose_concentration, bloom_state=Bloom.OPEN , color=random.choice(list(Color))):
        super().__init__(flower_id, bee_model)

        self.sucrose_concentration = sucrose_concentration

        #optional variables
        self.bloom_state = bloom_state
        self.color = color
