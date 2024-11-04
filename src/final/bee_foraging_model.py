#packages for modeling
import math
import mesa
from mesa.time import SimultaneousActivation

#packages for feature implementation
from enum import Enum
import random

#packages for data analysis
import pandas as pd
import numpy as np



### definition of fixed global variables ###
SUNRISE = 7 * 3600 # tick when the sun rises
SUNSET = 19 * 3600 # tick when the sun sets
TICKS_PER_DAY = 24 * 3600 # amount of ticks in a full day

RESTING_ENERGY_COST = 6.2 #mW
UNLOADING_NECTAR_ENERGY_COST = 9.3 #mW
FLYING_COST_UNLOADED = 37.0 #mW
FLYING_COST_LOADED = 75 #mW
FLYING_SPEED = 6.944 #m/s
SEARCHING_SPEED = FLYING_SPEED / 3 #m/s
SEARCHING_ANGLE_RANGE = (-60.0, 60.0) #°

MAX_SEARCH_TIME = 960 #s [or ticks]
MAX_DANCE_ERROR = 50 #m

NECTAR_REWARD = 353.8 #C


### definition of model specific parameters ###
GRID_RESOLUTION = 1
MAX_SIGHT = 1
MIN_BORDER_DISTANCE = 3

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


class FlowerAgent(mesa.Agent):
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
    def __init__(self, flower_id, bee_model, sucrose_concentration, flower_range = 1.0, bloom_state=Bloom.OPEN , color=random.choice(list(Color))):
        super().__init__(flower_id, bee_model)

        self.sucrose_concentration = sucrose_concentration

        #optional variables
        self.range = flower_range
        self.bloom_state = bloom_state
        self.color = color











class ForagerBeeAgent(mesa.Agent):
    """

        Represents a single forager bee in the model

        Args:
            bee_id (int): unique id of the bee agent
            bee_model (mesa.Model): model where the bee agent is placed
            days_of_experience (int): number of days of experience
            forager_type (BeeForagerType): type of bee agent
            start_pos ((float, float)): starting position of the bee agent

        Variables:
            target_pos ((float, float)): position the bee agent is heading to
            last_angle (float): flying angle of the bee agent at the last step

    """
    def __init__(self, bee_id, bee_model, forager_type, days_of_experience,  start_pos):
        super().__init__(bee_id, bee_model)

        self.forager_type = forager_type
        self.days_of_experience = days_of_experience
        self.accurate_position = start_pos



        self.target_pos = None
        self.last_angle = 0.0



    #todo: include energy loss function


    def search(self, destination):
        if get_distance(self.accurate_position, destination) <= MAX_SIGHT or self.last_move_crossed_flower_radius(self.last_angle, SEARCHING_SPEED):
            self.move_bee_towards_point(destination, SEARCHING_SPEED)

        elif self.model.grid.is_close_to_border(self.accurate_position):
            angle = (self.last_angle + math.pi) % (math.pi *2)
            self.move_bee_with_angle(angle, SEARCHING_SPEED)

        else:
            angle = (self.last_angle + random.uniform(math.radians(SEARCHING_ANGLE_RANGE[0]), math.radians(SEARCHING_ANGLE_RANGE[1]))) % (2 * math.pi)
            self.move_bee_with_angle(angle, SEARCHING_SPEED)



    def move_bee_towards_point(self, destination, speed):
        if self.model.grid.out_of_bounds(destination):
            raise ValueError("Destination out of bounds")

        angle = get_angle(self.accurate_position, destination)
        current_distance = math.sqrt((self.accurate_position[0] - destination[0]) ** 2 + (self.accurate_position[1] - destination[1]) ** 2)

        if current_distance < speed:
            self.accurate_position = destination
            self.model.grid.move_agent(self, (int(destination[0]), int(destination[1])))
            self.last_angle = angle

        else:
            new_pos = (self.accurate_position[0] + speed * math.cos(angle), self.accurate_position[1] + speed * math.sin(angle))
            if not self.model.grid.out_of_bounds(new_pos):
                self.accurate_position = new_pos
                self.model.grid.move_agent(self, (int(new_pos[0]), int(new_pos[1])))
                self.last_angle = angle




    def move_bee_with_angle(self, angle, speed):
        if not 0 <= angle <= math.pi * 2:
            raise ValueError("Angle out of range")

        new_pos = (self.accurate_position[0] + speed * math.cos(angle), self.accurate_position[1] + speed * math.sin(angle))

        if not self.model.grid.out_of_bounds(new_pos):
            self.accurate_position = new_pos
            self.model.grid.move_agent(self, (int(new_pos[0]), int(new_pos[1])))
            self.last_angle = angle


    def last_move_crossed_flower_radius(self, last_angle, speed):
        curr_x, curr_y = self.accurate_position
        last_x = self.accurate_position[0] - speed * math.cos(last_angle)
        last_y = self.accurate_position[1] - speed * math.sin(last_angle)

        return circle_line_intersect((last_x, last_y), (curr_x, curr_y), self.model.flower_location, MAX_SIGHT)




class BeeGrid(mesa.space.MultiGrid):
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

    def __init__(self, size, resolution):
        super().__init__(size * resolution, size * resolution, False)
        self.hive = (size * resolution // 2, size * resolution // 2)
        self.dance_floor = (self.hive[0] + 1, self.hive[1])


    def is_close_to_border(self, point):
        x,y = point
        return x < MIN_BORDER_DISTANCE or x > self.width - MIN_BORDER_DISTANCE or y < MIN_BORDER_DISTANCE or y > self.height - MIN_BORDER_DISTANCE



    def visualize_bee_grid(self):
        print()
        for y in range(self.height):
            for x in range(self.width):
                if (x,y) is self.dance_floor:
                    print("D", end="")
                elif (x,y) is self.hive:
                    print("H", end="")
                elif any(isinstance(flower, FlowerAgent) for flower in self.get_cell_list_contents([(x,y)])):
                    print("F", end="")
                elif any(isinstance(f_bee, ForagerBeeAgent) for f_bee in self.get_cell_list_contents([(x,y)])):
                    print("B", end="")
                else:
                    print("_", end="")
        print()





class BeeForagingModel(mesa.Model):
    """

        Bee Foraging Model

        Args:
            grid_size(int): size of the grid i.e. size x size
            grid_resolution(int): resolution of the grid



    """
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
        self.flower_range = get_surrounding(self.flower_location, MAX_SIGHT * GRID_RESOLUTION)




### model functions ###


def generate_random_point(origin_x, origin_y, target_distance, tolerance=0.1, max_attempts=10000):
    """
        Generates a random point with a given distance to the another points

        Args:
            origin_x (int): x coordinate of the origin point
            origin_y (int): y coordinate of the origin point
            target_distance (int): distance to the given coordinates
            tolerance (float): maximal deviation from the given distance

    """
    for _ in range(max_attempts):
        angle = random.uniform(0, 2* math.pi)

        # Add some randomness to the distance within the tolerance range
        min_distance = target_distance * (1 - tolerance)
        max_distance = target_distance * (1 + tolerance)
        actual_distance = random.uniform(min_distance, max_distance)

        # Calculate the point using polar coordinates
        x = origin_x + actual_distance * math.cos(angle)
        y = origin_y + actual_distance * math.sin(angle)

        # If both coordinates are positive, return the point
        if x >= 0 and y >= 0:
            return int(round(x, 2)), int(round(y, 2))



def get_surrounding(position, distance):
    """
        Returns a list of points that are within a given distance from the origin

        Args:
            position (int, int): center of the area
            distance (float): distance from the center of the area that's considered a surrounding
    """
    min_x, max_x = int(position[0] - 1.5*distance), int(position[0] + 1.5*distance)
    min_y, max_y = int(position[1] - 1.5*distance), int(position[1] + 1.5*distance)

    surrounding = []

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            if math.sqrt((x - position[0]) ** 2 + (y - position[1]) ** 2) <= distance:
                surrounding.append((x, y))

    return surrounding



def get_next_point(current_x, current_y, angle, distance):
    """
        Returns the next point given a start point, direction(angle) and distance

        Args:
            current_x (int): x coordinate of the current point
            current_y (int): y coordinate of the current point
            angle (float): angle between 0 and 2pi that determines the direction of movement
            distance (float): distance that's to be covered
    """
    x_next = current_x + distance * math.cos(angle)
    y_next = current_y + distance * math.sin(angle)
    return round(x_next, 2), round(y_next, 2)



def get_distance(pos1, pos2):
    """
        Uses euclidian distance
    """
    return round(math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2), 2)





def get_angle(starting_point, destination_point):
    """
        the angle in a triangle is calculated by arctan(y/x)

        We use this principle to calculate the angle/direction in which an object would have to
        be moved to reach a given point
    """

    dx = destination_point[0] - starting_point[0]
    dy = destination_point[1] - starting_point[1]

    angle = math.atan2(dy, dx)

    # Convert from [-π, π] to [0, 2π]
    if angle < 0:
        angle += 2 * math.pi

    return angle



def radians_to_degrees(radians):
    return radians * (180 / math.pi)


def circle_line_intersect(p1, p2, circle_center, radius):
    # Extract coordinates
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = circle_center

    # Vector from p1 to p2
    dx = x2 - x1
    dy = y2 - y1

    # Vector from p1 to circle center
    pcx = cx - x1
    pcy = cy - y1

    # Length of line segment squared
    line_length_sq = dx * dx + dy * dy

    # Skip if line segment has zero length
    if line_length_sq == 0:
        # Check if p1 is within circle
        return math.sqrt(pcx * pcx + pcy * pcy) <= radius

    # Project circle center onto line segment
    proj = (pcx * dx + pcy * dy) / line_length_sq

    # Find closest point on line segment to circle center
    if proj < 0:
        closest_x, closest_y = x1, y1
    elif proj > 1:
        closest_x, closest_y = x2, y2
    else:
        closest_x = x1 + proj * dx
        closest_y = y1 + proj * dy

    # Calculate distance from closest point to circle center
    distance = math.sqrt(
        (closest_x - cx) * (closest_x - cx) +
        (closest_y - cy) * (closest_y - cy)
    )

    # Compare with radius
    return distance <= radius



succ = 0

for _ in range(500):
    run_model = BeeForagingModel(900)

    bee = ForagerBeeAgent(1, run_model, BeeForagerType.PERSISTENT, 1, run_model.grid.hive)
    run_model.grid.place_agent(bee, run_model.grid.hive)

    target = run_model.flower_location
    dance_location = (run_model.flower_location[0]-5, run_model.flower_location[1]+5)

    bee.target_pos = dance_location


    while bee.accurate_position is not dance_location:
        bee.move_bee_towards_point(bee.target_pos, FLYING_SPEED)


    for i in range(960):
        if bee.accurate_position is target:
            succ += 1
            break


    print(run_model.flower_location)



print(succ)
