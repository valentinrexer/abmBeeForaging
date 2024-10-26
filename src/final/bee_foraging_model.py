#packages for modeling
import math

import mesa

#packages for feature implementation
from enum import Enum
import random

#packages for data analysis
import pandas as pd
import numpy as np
from matplotlib.mlab import window_none
from mesa.time import SimultaneousActivation

### definition of fixed global variables ###
SUNRISE = 25,200 # tick when the sun rises
SUNSET = 68,400 # tick when the sun sets
TICKS_PER_DAY = 86,400 # amount of ticks at a full day

RESTING_ENERGY_COST = 6.2 #mW
UNLOADING_NECTAR_ENERGY_COST = 9.3 #mW
FLYING_COST_UNLOADED = 37.0 #mW
FLYING_COST_LOADED = 75 #mW
FLYING_SPEED = 6.944 #m/s
SEARCHING_SPEED = FLYING_SPEED / 3 #m/s
MAX_SEARCH_TIME = 960 #s
MAX_DANCE_ERROR = 50 #m

NECTAR_REWARD = 353.8 #C


### definition of model specific parameters ###
GRID_RESOLUTION = 10
FLOWER_SURROUNDING = 1


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
        
        flower_range (float): 
        closed (bool) [optional]: flower is closed
        color (Color) [optional]: color of the flower
"""

class FlowerAgent(mesa.Agent):
    def __init__(self, flower_id, bee_model, sucrose_concentration, flower_range = 1.0, bloom_state=Bloom.OPEN , color=random.choice(list(Color))):
        super().__init__(flower_id, bee_model)

        self.sucrose_concentration = sucrose_concentration

        #optional variables
        self.range = flower_range
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



"""
    
    Bee Foraging Model
    
    Args:
        grid_size(int): size of the grid i.e. size x size
        grid_resolution(int): resolution of the grid
        
        
        
"""

class BeeForagingModel(mesa.Model):
    def __init__(self, source_distance):
        super().__init__()

        # create schedule
        self.schedule = SimultaneousActivation(self)

        # create grid ==> grid will have a size of source_distance + 70
        self.grid = BeeGrid((source_distance + 70) * 2, GRID_RESOLUTION)

        #create flower (food source) and place it on the grid
        #todo: ask for actual sucrose concentration
        flower = FlowerAgent(1, self, 1000)
        self.schedule.add(flower)
        self.flower_location = generate_random_point(self.grid.hive[0], self.grid.hive[1], source_distance * GRID_RESOLUTION, 0.01)
        self.flower_range = get_surrounding(self.flower_location, FLOWER_SURROUNDING * GRID_RESOLUTION)




### model functions ###

"""
    Generates a random point with a given distance to the another points
    
    Args:
        origin_x (int): x coordinate of the origin point
        origin_y (int): y coordinate of the origin point
        target_distance (int): distance to the given coordinates
        tolerance (float): maximal deviation from the given distance
    
"""
def generate_random_point(origin_x, origin_y, target_distance, tolerance=0.1):

    while True:
        angle = random.uniform(0, math.pi / 2)

        # Add some randomness to the distance within the tolerance range
        min_distance = target_distance * (1 - tolerance)
        max_distance = target_distance * (1 + tolerance)
        actual_distance = random.uniform(min_distance, max_distance)

        # Calculate the point using polar coordinates
        x = origin_x + actual_distance * math.cos(angle)
        y = origin_y + actual_distance * math.sin(angle)

        # If both coordinates are positive, return the point
        if x >= 0 and y >= 0:
            print(math.sqrt((origin_x - x) ** 2 + (origin_y - y) ** 2))
            return int(round(x, 2)), int(round(y, 2))



def get_surrounding(position, distance):
    min_x, max_x = int(position[0] - 1.5*distance), int(position[0] + 1.5*distance)
    min_y, max_y = int(position[1] - 1.5*distance), int(position[1] + 1.5*distance)

    surrounding = []

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            if math.sqrt((x - position[0]) ** 2 + (y - position[1]) ** 2) <= distance:
                surrounding.append((x, y))

    return surrounding



run_model = BeeForagingModel(100)
print(run_model.grid.width)
print(run_model.grid.height)
print(run_model.flower_location)
print(len(run_model.flower_range))






