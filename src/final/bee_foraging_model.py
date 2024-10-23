#packages for modeling
import mesa

#packages for feature implementation
from enum import Enum
import random

#packages for data analysis




### definition of global attributes ###

#flower colors
class Color(Enum):
    RED = 1,
    GREEN = 2,
    BLUE = 3,
    PURPLE = 4,
    YELLOW = 5,
    WHITE = 6,


#bee status



# definition of the Flower Class
#todo: adjust flower stock variable
"""
    Represents a single flower in the model 
    
    Args:
        flower_id (int): unique id of the flower agent
        bee_model (mesa.Model): model where the agent is placed
        
        closed (bool) [optional]: flower is closed
        flower_stock (float) [optional]: nectar stock / energy? 
        color (Color) [optional]: color of the flower
"""
class FlowerAgent(mesa.Agent):
    def __init__(self, flower_id, bee_model, closed=False,  flower_stock = random.randint(0, 100) / 100, color=random.choice(list(Color))):
        super().__init__(flower_id, bee_model)


        #optional variables
        self.closed = closed
        self.flower_stock = flower_stock
        self.color = color
