#packages for modeling
import mesa

#packages for feature implementation
from enum import Enum
import random

#packages for data analysis
import pandas as pd
import numpy as np


### definition of fixed global variables ###

RESTING_ENERGY_COST = 6.2 #mW
UNLOADING_NECTAR_ENERGY_COST = 9.3 #mW
FLYING_COST_UNLOADED = 37.0 #mW
FLYING_COST_LOADED = 75 #mW
FLYING_SPEED = 6.944 #m/s
SEARCHING_SPEED = FLYING_SPEED / 3 #m/s
MAX_SEARCH_TIME = 960 #s
MAX_DANCE_ERROR = 50 #m

NECTAR_REWARD = 353.8 #C



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


#todo: will probably be replaced/removed later (experience status might be retrievable from days known variable)
#bee forager experienced status
class BeeForagerExperienceStatus(Enum):
    EXPERIENCED_FORAGER = 1,
    NEW_FORAGER = 2,


#bee forager type
class BeeForagerType(Enum):
    PERSISTENT = 1,
    RETICENT = 2,


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


#foraging strategy
class ForagingStrategy(Enum):
    STRATEGY_1 = 1,
    STRATEGY_10 = 2,
    STRATEGY_100 = 3,
    STRATEGY_40_60_80 = 4,




# definition of the Flower Class
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








"""

    Represents a single forager bee in the model
    
    Args:
        bee_id (int): unique id of the bee agent
        bee_model (mesa.Model): model where the bee agent is placed
        
"""
class ForagerBeeAgent(mesa.Agent):
    def __init__(self, bee_id, bee_model):
        super().__init__(bee_id, bee_model)








"""

    Custom Grid to model the environment for the simulation
    
    Args:
        size (int): size of the grid i.e. size x size
        resolution (int): resolution of the grid (e.g. resolution=10 means 1m = 10 units size units in the model)
    
    Variables:
        hive ( (int,int) ): position of the hive ==> bee agents are located here while they rest or unload, 
                            located at the center of the grid
                            
        dance_floor ( (int,int) ): position of the dance floor ==> bee agents are located here while they cluster or dance
                                   located at (hive_x+1, hive_y)
"""


class BeeGrid(mesa.space.MultiGrid):

    def __init__(self, size, resolution):
        super().__init__(size * resolution, size * resolution, False)
        self.hive = (size * resolution // 2, size * resolution // 2)
        self.dance_floor = (self.hive[0] + 1, self.hive[1])
