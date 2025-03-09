#packages for modeling
import math
import sys
import warnings

import mesa

#packages for feature implementation
from enum import Enum
import random

#packages for multicore processing
import multiprocessing as mp

#packages for data analysis
import pandas as pd
import numpy as np
from pandas.core.roperator import rand_

### definition of fixed global variables ###
SUNRISE = 7 * 3600 # tick when the sun rises
SUNSET = 19 * 3600 # tick when the sun sets
STEPS_PER_DAY = 24 * 3600 # amount of ticks in a full day
NUMBER_OF_DAYS = 5 # number of simulated days per simulation

RESTING_ENERGY_COST = 6.2 * (1.0/1000) #mJ/s
UNLOADING_NECTAR_ENERGY_COST = 9.3 * (1.0/1000) #mJ/s
FLYING_COST_UNLOADED = 37.0 * (1.0/1000) #mJ/s
FLYING_COST_LOADED = 75 * (1.0/1000) #mJ/s
FLYING_SPEED = 6.944 #m/s
SEARCHING_SPEED = FLYING_SPEED / 3 #m/s
SEARCHING_ANGLE_RANGE = (-120.0, 120.0) #°

MAX_SEARCH_TIME = 960 #s [or ticks]
MAX_DANCE_ERROR = 30 #m

NECTAR_REWARD = 353.8 #*C J


### definition of model specific parameters ###
MAX_SIGHT = 1
MIN_BORDER_DISTANCE = 3
MAX_SEARCH_RADIUS = 50  #m max radius in which bee looks for the source


### definition of normal distribution variables for random drawing ###
MEAN = 0
STANDARD_DEVIATION = 35


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

#bee status
class BeeStatus(Enum):
    RESTING = 1,
    CLUSTERING = 2,
    DANCING = 3,
    WATCHING_WAGGLE_DANCE = 4,
    PREPARING_TO_FLY_OUT = 5,
    RETURNING = 6,
    LOADING_NECTAR = 7,
    UNLOADING_NECTAR = 8,
    FLYING_TO_SEARCH_AREA = 9,
    SEARCHING_ADVERTISED_SOURCE = 10,
    FLYING_STRAIGHT_TO_FLOWER = 11,



class ForagingStrategy(Enum):
    PERSISTENT = 1,
    RETICENT = 2,


# definition of the Flower Class
class FlowerAgent(mesa.Agent):
    """
        Represents a single flower in the model

        Args:
            flower_id (int): unique id of the flower agent
            bee_model (mesa.Model): model where the agent is placed
            sucrose_stock (float): available nectar / sucrose in the flower

            flower_range (float):
            closed (bool) [optional]: flower is closed
            color (Color) [optional]: color of the flower
    """
    def __init__(self, bee_model, location, sucrose_stock, open_time, close_time,  visibility_radius = 1.0, bloom_state=Bloom.CLOSED , color=random.choice(list(Color))):
        if not isinstance(bee_model, BeeForagingModel):
            raise TypeError("bee_model must be of type BeeForagingModel")

        super().__init__(bee_model)

        self.sucrose_stock = sucrose_stock   # available sucrose in flower agent
        self.open_time = open_time   # tick/step when the flower opens ==> bee can access the flower
        self.close_time = close_time   # tick/step when the flower closes ==> no bee can access the flower
        self.location = location   # location of the flower
        self.distance_from_hive = get_distance(self.location, bee_model.hive)

        C = bee_model.sucrose_concentration
        D = self.distance_from_hive
        self.value = (353.6 * C - 0.015 * D - 2.64) / (0.2 * C + 0.015 * D + 2.64)

        #optional variables
        self.visibility_radius = visibility_radius
        self.bloom_state = bloom_state
        self.color = color


    def step(self):
        if self.model.steps == self.open_time % STEPS_PER_DAY:
            self.bloom_state = Bloom.OPEN

        if self.model.steps == self.close_time % STEPS_PER_DAY:
            self.bloom_state = Bloom.CLOSED

class ForagerBeeAgent(mesa.Agent):
    """

        Represents a single forager bee in the model

        Args:
            bee_model (mesa.Model): model where the bee agent is placed
            days_of_experience (int): number of days of experience
            start_pos ((float, float)): starting position of the bee agent

        Variables:
            target_pos ((float, float)): position the bee agent is heading to
            last_angle (float): flying angle of the bee agent at the last step

    """
    def __init__(self, bee_model, days_of_experience, start_pos, time_source_found=-1):
        super().__init__(bee_model)

        self.days_of_experience = days_of_experience
        self.accurate_position = start_pos
        self.status = BeeStatus.RESTING
        self.remaining_time_in_state = 0
        self.time_source_found = time_source_found
        self.last_collection_time = None
        self.next_anticipation_time = None
        self.foraging_strategy = None
        self.next_reconnaissance_time = None
        self.loaded = False
        self.collection_times = []

        self.targeted_flower = None
        self.last_angle = 0.0
        self.search_area_center = None
        self.currently_watched_bee = None
        self.homing_motivation = 0
        self.sucrose_load = 0

    def anticipation(self, method , sunrise, sunset):
        """
        Simulating the bees anticipation behaviour based on her experience

        :param method: Anticipation method
        :param sunrise: sunrise time for this simulation
        :param sunset: sunset time for this simulation
        :return: anticipated time for source
        """
        if not isinstance(self.model, BeeForagingModel):
            raise TypeError("bee_model must be of type BeeForagingModel")

        time_source_found = self.time_source_found % STEPS_PER_DAY
        distance = get_distance(self.targeted_flower.location, self.model.hive)

        if not sunrise <= time_source_found <= sunset:
            warnings.warn("Invalid argument for time_source_found")

        if method == 1:
            anticipation = time_source_found - distance/FLYING_SPEED
            return int(anticipation + self.model.get_current_day() * STEPS_PER_DAY)

        elif method == 2:
            anticipation = (time_source_found - 3600 * 4 * (time_source_found - sunrise) / (sunset - sunrise)) - distance/FLYING_SPEED
            return int(anticipation + self.model.get_current_day() * STEPS_PER_DAY)

        else:
            return -1

    def search(self, flower, search_area_center):
        """
        Models the bee search behaviour
            => If the bee is closer than MAX_SIGHT  to the source it or was in this range sometimes between the current and last time step it directly flies to the flower
            => If the bee

        :param flower: the position the bee tries to find
        :param search_area_center: a tuple that describes the area in which the bee tries to find the destination
        """
        self.search_area_center = search_area_center

        if get_distance(self.accurate_position, flower.location) <= flower.visibility_radius or self.last_move_crossed_flower_radius(self.last_angle, SEARCHING_SPEED, flower):
            self.move_bee_towards_point(flower.location, SEARCHING_SPEED)

        elif get_distance(self.accurate_position, self.search_area_center) >= MAX_SEARCH_RADIUS:
            angle = get_angle(self.accurate_position, self.search_area_center)
            angle = random_deviate_angle_equally(angle, SEARCHING_ANGLE_RANGE[0], SEARCHING_ANGLE_RANGE[1])
            self.move_bee_with_angle(angle, SEARCHING_SPEED)

        else:
            angle = random_deviate_angle(self.last_angle, MEAN, STANDARD_DEVIATION, SEARCHING_ANGLE_RANGE[0], SEARCHING_ANGLE_RANGE[1])
            self.move_bee_with_angle(angle, SEARCHING_SPEED)

        self.homing_motivation += 1

    def move_bee_towards_point(self, destination, speed):

        """
        Moves a bee agent towards a point. On tick in this model equals one second in ==> The distance the bee travels
        equals her speed in m/s

        :param destination: the position the bee travels towards
        :param speed: the flight speed of the bee
        """
        if not isinstance(self.model, BeeForagingModel):
            raise TypeError("bee_model must be of type BeeForagingModel")

        if self.model.out_of_bounds(destination):
            raise ValueError("Destination out of bounds")

        angle = get_angle(self.accurate_position, destination)
        current_distance = math.sqrt((self.accurate_position[0] - destination[0]) ** 2 + (self.accurate_position[1] - destination[1]) ** 2)

        if current_distance < speed:
            self.accurate_position = destination
            self.last_angle = angle

        else:
            new_pos = (self.accurate_position[0] + speed * math.cos(angle), self.accurate_position[1] + speed * math.sin(angle))
            if not self.model.out_of_bounds(new_pos):
                self.accurate_position = new_pos
                self.last_angle = angle

    def move_bee_with_angle(self, angle, speed):
        """
        Moving a bee on the grid with a certain angle and a certain speed. Each direction on the grid can be represented
        by a number between 0 and 2 * pi

        :param angle: the angle representing the direction of the bee
        :param speed: the flight speed of the bee
        :return:
        """

        if not isinstance(self.model, BeeForagingModel):
            raise TypeError("bee_model must be of type BeeForagingModel")

        if not 0 <= angle <= math.pi * 2:
            raise ValueError("Angle out of range")

        new_pos = (self.accurate_position[0] + speed * math.cos(angle), self.accurate_position[1] + speed * math.sin(angle))

        if not self.model.out_of_bounds(new_pos):
            self.accurate_position = new_pos
            self.last_angle = angle

    def last_move_crossed_flower_radius(self, last_angle, speed, flower):
        """
        If the bee crosses the radius in which she's able to see the flower between to tick, but she still ends
        up outside the circle, we assume she saw the flower when she crossed the circle ==> the bee is placed on the
        flower. Here we check if the bee crossed the circle during the last time step

        :param flower: targeted flower
        :param last_angle: the last flying direction of the bee
        :param speed: the bees flying speed
        :return: boolean if the bee crossed the flower radius or not
        """

        if not isinstance(flower, FlowerAgent):
            raise ValueError("flower must be a FlowerAgent")

        curr_x, curr_y = self.accurate_position
        last_x = self.accurate_position[0] - speed * math.cos(last_angle)
        last_y = self.accurate_position[1] - speed * math.sin(last_angle)

        return circle_line_intersect((last_x, last_y), (curr_x, curr_y), flower.location, flower.visibility_radius)

    def load_nectar(self, flower):
        if self.remaining_time_in_state > 0:
            self.remaining_time_in_state -= 1

        else:
            C = self.model.sucrose_concentration
            self.sucrose_load += NECTAR_REWARD * C
            flower.sucrose_stock -= NECTAR_REWARD * C
    
    def update_status(self):
        """
        Updates the current status of the bee after each time step
        """
        if not isinstance(self.model, BeeForagingModel):
            raise ValueError("The bee foraging model must be a BeeForagingModel")


        if self.status == BeeStatus.RESTING:
            if self.model.steps % STEPS_PER_DAY == self.next_anticipation_time % STEPS_PER_DAY:
                self.status = BeeStatus.CLUSTERING

                if self.foraging_strategy == ForagingStrategy.PERSISTENT:
                    self.next_reconnaissance_time = random.randint(self.next_anticipation_time, (self.time_source_found % STEPS_PER_DAY) + (self.model.get_current_day()) * STEPS_PER_DAY)


        elif self.status == BeeStatus.CLUSTERING:
            if self.currently_watched_bee is not None and (self.targeted_flower is self.currently_watched_bee.targeted_flower or self.targeted_flower is None):
                self.status = BeeStatus.WATCHING_WAGGLE_DANCE
                self.targeted_flower = self.currently_watched_bee.targeted_flower if self.targeted_flower is None else self.targeted_flower
                self.remaining_time_in_state = int(((0.0013 * self.targeted_flower.distance_from_hive) + 1.86) * random.uniform(4.9, 12.9))

            elif self.foraging_strategy == ForagingStrategy.PERSISTENT and self.next_reconnaissance_time == self.model.steps:
                self.status = BeeStatus.FLYING_STRAIGHT_TO_FLOWER

            elif self.foraging_strategy == ForagingStrategy.RETICENT and self.last_collection_time % STEPS_PER_DAY == self.model.steps % STEPS_PER_DAY:
                try:
                    self.status = BeeStatus.RESTING
                    self.last_collection_time = self.collection_times[-1]

                except IndexError:
                    print(self.last_collection_time)


        elif self.status == BeeStatus.WATCHING_WAGGLE_DANCE:
            if self.remaining_time_in_state > 0:
                self.remaining_time_in_state -= 1

            else:
                if self.days_of_experience >= 0:
                    self.status = BeeStatus.FLYING_STRAIGHT_TO_FLOWER

                else:
                    self.status = BeeStatus.FLYING_TO_SEARCH_AREA
                    rand_distance_from_flower = random.randint(1,MAX_SEARCH_RADIUS)
                    self.search_area_center = generate_random_point(self.targeted_flower.location[0], self.targeted_flower.location[1], rand_distance_from_flower)


                self.currently_watched_bee = None


        elif self.status == BeeStatus.FLYING_STRAIGHT_TO_FLOWER or self.status == BeeStatus.SEARCHING_ADVERTISED_SOURCE:
            if self.accurate_position == self.targeted_flower.location:
                if self.targeted_flower.bloom_state == Bloom.OPEN:
                    self.status = BeeStatus.LOADING_NECTAR
                    self.collection_times.append(self.model.steps)

                    C = self.model.sucrose_concentration
                    self.remaining_time_in_state = int(random.uniform(12.44 * C + 20.09, 24.22 * C + 32.07))
                    self.time_source_found = self.model.steps if (self.time_source_found // STEPS_PER_DAY) != (self.model.get_current_day() - 1) else self.time_source_found

                else:
                    self.status = BeeStatus.RETURNING

                    if self.foraging_strategy == ForagingStrategy.PERSISTENT:
                        self.next_reconnaissance_time = int((self.next_reconnaissance_time + self.time_source_found) / 2) if abs(self.next_reconnaissance_time - self.time_source_found) > 120 else self.model.steps + 120

            elif self.homing_motivation > MAX_SEARCH_TIME:
                self.status = BeeStatus.RETURNING
                self.homing_motivation = 0


        elif self.status == BeeStatus.FLYING_TO_SEARCH_AREA:
            if self.accurate_position == self.search_area_center:
                self.status = BeeStatus.SEARCHING_ADVERTISED_SOURCE


        elif self.status == BeeStatus.LOADING_NECTAR:
            if self.remaining_time_in_state > 0:
                self.remaining_time_in_state -= 1

            else:
                self.loaded = True
                self.status = BeeStatus.RETURNING


        elif self.status == BeeStatus.RETURNING:
            if self.accurate_position == self.model.dance_floor:
                self.status = BeeStatus.UNLOADING_NECTAR
                C = self.model.sucrose_concentration
                self.remaining_time_in_state = int(random.uniform(39 * (C ** 2) + 114.1 * C - 64.25, 159 * (C ** 2) - 140 * C + 166))


        elif self.status == BeeStatus.UNLOADING_NECTAR:
            if self.remaining_time_in_state > 0:
                self.remaining_time_in_state -= 1

            else:
                self.model.total_energy += NECTAR_REWARD * self.model.sucrose_concentration
                self.loaded = False
                self.status = BeeStatus.DANCING
                self.remaining_time_in_state = 0.1713 * self.targeted_flower.value


        elif self.status == BeeStatus.DANCING:
            if self.remaining_time_in_state > 0:
                self.remaining_time_in_state -= 1

            else:
                self.status = BeeStatus.PREPARING_TO_FLY_OUT
                self.remaining_time_in_state = random.randint(16, 51)


        elif self.status == BeeStatus.PREPARING_TO_FLY_OUT:
            if self.remaining_time_in_state > 0:
                self.remaining_time_in_state -= 1

            else:
                self.status = BeeStatus.FLYING_STRAIGHT_TO_FLOWER

        else:
            self.status = self.status



    def step(self):
        """
        Step function of the bee
        """
        if not isinstance(self.model, BeeForagingModel):
            raise AttributeError("BeeForagingModel is not an instance of BeeForagingModel")

        self.update_status()

        if (self.status == BeeStatus.RESTING or
                self.status == BeeStatus.WATCHING_WAGGLE_DANCE or
                self.status == BeeStatus.UNLOADING_NECTAR or
                self.status == BeeStatus.LOADING_NECTAR or
                self.status == BeeStatus.DANCING or
                self.status == BeeStatus.PREPARING_TO_FLY_OUT):

            self.model.total_energy -= RESTING_ENERGY_COST

        elif self.status == BeeStatus.CLUSTERING:
            self.model.total_energy -= RESTING_ENERGY_COST
            seen_bees = split_agents_by_percentage(self.model.get_bees_on_dance_floor(), 4)[0]

            for bee in seen_bees:
                if bee.status == BeeStatus.DANCING and bee.targeted_flower is self.targeted_flower:
                    self.currently_watched_bee = bee
                    break


        elif self.status == BeeStatus.FLYING_STRAIGHT_TO_FLOWER:
            self.model.total_energy -= FLYING_COST_UNLOADED

            if get_distance(self.accurate_position, self.targeted_flower.location) <= FLYING_SPEED:
                self.accurate_position = self.targeted_flower.location

            else:
                self.move_bee_towards_point(self.targeted_flower.location, FLYING_SPEED)


        elif self.status == BeeStatus.FLYING_TO_SEARCH_AREA:
            self.model.total_energy -= FLYING_COST_UNLOADED
            if get_distance(self.accurate_position, self.search_area_center) <= FLYING_SPEED:
                self.accurate_position = self.search_area_center


        elif self.status == BeeStatus.SEARCHING_ADVERTISED_SOURCE:
            self.search(self.targeted_flower.location, self.search_area_center)


        elif self.status == BeeStatus.RETURNING:
            if self.loaded:
                self.model.total_energy -= FLYING_COST_LOADED

            else:
                self.model.total_energy -= FLYING_COST_UNLOADED


            if get_distance(self.accurate_position, self.model.dance_floor) <= FLYING_SPEED:
                self.accurate_position = self.model.dance_floor

            else:
                self.move_bee_towards_point(self.model.dance_floor, FLYING_SPEED)











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

    def __init__(self, size):
        super().__init__(size, size, False)
        self.hive = (size // 2, size // 2)
        self.dance_floor = (self.hive[0] + 1, self.hive[1])


    def is_close_to_border(self, point):
        """
        checks if a position on the grid is too close to the border of the grid

        :param point: location to be tested in tuple format
        :return: boolean if the position on the grid is too close to the border of the grid
        """
        x, y = point
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

    def get_all_dancing_bees(self):
        return [bee for bee in self.get_cell_list_contents([self.dance_floor]) if isinstance(bee, ForagerBeeAgent) and bee.BeeStatus == BeeStatus.DANCING]


#todo: adjust class so flower location is set
class BeeForagingModel(mesa.Model):
    """

        Bee Foraging Model

        Args:
            grid_size(int): size of the grid i.e. size x size
            grid_resolution(int): resolution of the grid



    """
    def __init__(self, source_distance, number_of_starting_bees, sucrose_concentration=1):
        super().__init__()

        # initialize model_information
        self.number_of_starting_bees = number_of_starting_bees
        self.sucrose_concentration = sucrose_concentration
        self.total_energy = 0


        self.agents.do("step")
        self.agents.do("advance")

        # create grid ==> grid will have a size of source_distance + 70 in both directions
        self.size = (source_distance + 70) * 2
        self.hive = (self.size // 2, self.size // 2)
        self.dance_floor = (self.hive[0] + 1, self.hive[1])

        #create flower (food source) and place it on the grid
        #todo: ask for actual sucrose concentration
        #todo: in the future automize generation of multiple flowers
        flower_location = generate_random_point(self.hive[0], self.hive[1], source_distance, 0.01)
        flower = FlowerAgent(self, flower_location, 1000, 9 * 3600, 14 * 3600)
        self.agents.add(flower)


        #initialize all simulations specific variables that are passed as command line arguments
        self.sucrose_concentration = sucrose_concentration

        #variables for collecting results and scores
        self.collected_sucrose_rewards = 0.0


        # add the desired number of Bee Agents to the grid
        for i in range(number_of_starting_bees):
            bee_agent = ForagerBeeAgent(self, 1, (self.hive[0], self.hive[1]), time_source_found=random.randint(flower.open_time, flower.close_time))
            bee_agent.targeted_flower = flower
            bee_agent.next_anticipation_time = bee_agent.anticipation(1, SUNRISE, SUNSET)
            bee_agent.last_collection_time = random.randint(bee_agent.time_source_found, flower.close_time)
            self.agents.add(bee_agent)

        self.update_bee_foraging_strategies()



    def start(self):
        """
        prepares the model to start a simulation
        :return:
        """

    def debug(self, steps):
        for _ in range(steps):
            self.step()
            for bee in self.agents:
                if not isinstance(bee, ForagerBeeAgent):
                    continue



    def run(self, steps):
        """
        runs the model for an arbitrary number of steps
        :param steps:
        :return:
        """
        for _ in range(steps):
            self.step()

    def out_of_bounds(self, pos) -> bool:
        """Determines whether position is off the grid, returns the out of bounds coordinate."""
        x, y = pos
        return x < 0 or x >= self.size or y < 0 or y >= self.size

    def get_bees(self):
        return [bee for bee in self.agents if isinstance(bee, ForagerBeeAgent)]

    def get_bees_on_dance_floor(self):
        return [bee for bee in self.get_bees() if bee.accurate_position == self.dance_floor]

    def get_flowers(self):
        return [flower for flower in self.agents if isinstance(flower, FlowerAgent)]

    def get_current_day(self):
        return self.steps // STEPS_PER_DAY

    def update_bee_foraging_strategies(self):
        self.reassign_foraging_strategies(1, 40)
        self.reassign_foraging_strategies(2, 60)
        self.reassign_foraging_strategies(3, 80)
        self.reassign_foraging_strategies(4, 90)
        self.reassign_foraging_strategies(5, 95)


    def reassign_foraging_strategies(self, days_of_experience, persistent_percentage):
        bees = [bee for bee in self.agents if isinstance(bee, ForagerBeeAgent) and bee.days_of_experience == days_of_experience]
        bee_groups = split_agents_by_percentage(bees, persistent_percentage)

        for bee in bee_groups[0]:
            bee.foraging_strategy = ForagingStrategy.PERSISTENT

        for bee in bee_groups[1]:
            bee.foraging_strategy = ForagingStrategy.RETICENT


    def add_new_foragers(self, number_of_new_foragers):
        for _ in range(number_of_new_foragers):
            new_bee_agent = ForagerBeeAgent(self, 0, (self.hive[0], self.hive[1]))
            new_bee_agent.next_anticipation_time = SUNRISE + (self.get_current_day()) * STEPS_PER_DAY
            new_bee_agent.foraging_strategy = ForagingStrategy.RETICENT
            self.agents.add(new_bee_agent)

    def kill_agents(self, percentage = 0.2):
        # Get all bee agents
        bee_agents = [agent for agent in self.agents if isinstance(agent, ForagerBeeAgent)]

        # Calculate how many to kill
        num_to_kill = int(len(bee_agents) * percentage)

        # Select which bees to kill randomly
        bees_to_kill = random.sample(bee_agents, num_to_kill)

        # Remove these bees from the model
        for bee in bees_to_kill:
            self.agents.remove(bee)


    def daily_update(self):
        for bee in self.agents:
            if not isinstance(bee, ForagerBeeAgent):
                continue

            if bee.last_collection_time is not None:
                bee.days_of_experience += 1


    def step(self) -> None:
        self.agents.do("step")


### model functions ###
def generate_random_point(origin_x, origin_y, target_distance, tolerance=0.1, max_attempts=10000):
    """
    Generate a random point on the grid with a given distance to a given point (the hive in our model)

    :param origin_x: x coordinate of the given point
    :param origin_y: y coordinate of the given point
    :param target_distance: the distance between the given point and the new point
    :param tolerance: max difference between the desired and actual distance between the given point and the new point
    :param max_attempts: max number of attempts to find a point on the grid that fulfills all criteria (to prevent an infinity loop)
    :return:
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
        Use euclidian distance to calculate the distance between two points
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

    return normalize_angle(angle)

def random_deviate_angle(current_angle, mean, standard_deviation, min_value, max_value, radians=False):
    """
    Alters a given angle by a randomly drawn value

    :param current_angle:
    :param mean:
    :param standard_deviation:
    :param min_value:
    :param max_value:
    :param radians:
    :return:
    """
    deviation = draw_normal_distributed_value(mean, standard_deviation, min_value, max_value)

    if not radians:
        deviation = math.radians(deviation)

    current_angle += deviation
    current_angle = current_angle % (2 * math.pi)

    if current_angle < 0:
        current_angle += 2 * math.pi

    return current_angle

def random_deviate_angle_equally(angle, min_value, max_value, radians=False):
    deviation = random.uniform(min_value, max_value)

    if radians:
        angle += deviation

    else:
        angle += math.radians(deviation)

    return normalize_angle(angle)

def normalize_angle(current_angle):
    angle = current_angle % (2 * math.pi)

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

def draw_normal_distributed_value(mean, standard_deviation, min_value, max_value):
    while True:
        value = random.normalvariate(mean, standard_deviation)
        if min_value <= value <= max_value:
            return value

def get_day_of_step(step):
    return (step // STEPS_PER_DAY) + 1

def split_agents_by_percentage(agents, first_percentage=30):
    """
    Split agents of a specific class into two groups randomly based on a percentage split.

    Args:
        model: The Mesa model instance containing agents
        agent_class: The class of agents to filter and split (e.g., ForagerBeeAgent)
        first_percentage (int): Percentage of agents to include in the first group (0-100)
                               The second group gets (100 - first_percentage)

    Returns:
        tuple: A tuple containing (first_group_agents, second_group_agents)
    """

    if not agents:
        return [], []

    # Ensure percentage is within bounds
    first_percentage = max(0, min(100, first_percentage))

    # Calculate how many agents go in the first group
    num_in_first = int(len(agents) * (first_percentage / 100))

    # Create a copy and shuffle it
    shuffled_agents = agents.copy()
    random.shuffle(shuffled_agents)

    # Split the shuffled agents
    first_group = shuffled_agents[:num_in_first]
    second_group = shuffled_agents[num_in_first:]

    return first_group, second_group

def start_simulation(model):
    for bee_agent in [agent for agent in model.agents if isinstance(agent, ForagerBeeAgent)]:
        bee_agent.status = BeeStatus.FLYING_TO_SEARCH_AREA
        


def run_bee_model_instance(params):

    distance, foragers, sd = params

    model_instance = BeeForagingModel(int(distance))

    for k in range(int(foragers)):
        bee_agent = ForagerBeeAgent(k, model_instance, 1, model_instance.grid.hive)
        bee_agent.status = BeeStatus.FLYING_TO_SEARCH_AREA
        model_instance.schedule.add(bee_agent)

        inc_X = random.randint(-35,35)
        inc_Y = random.randint(-35,35)
        bee_agent.targeted_position = (model_instance.flower_location[0]+inc_X, model_instance.flower_location[1]+inc_Y)
        bee_agent.search_area_center = (model_instance.flower_location[0]+inc_X, model_instance.flower_location[1]+inc_Y)


        model_instance.grid.place_agent(bee_agent, model_instance.grid.hive)

    found_at_day = []

    it = 0
    c = 0


    while True:
        it += 1

        for forager in model_instance.schedule.agents:
            if isinstance(forager, ForagerBeeAgent):
                if forager.status == BeeStatus.FLYING_TO_SEARCH_AREA or forager.status == BeeStatus.SEARCHING_ADVERTISED_SOURCE:
                    forager.status = BeeStatus.FLYING_TO_SEARCH_AREA
                    forager.accurate_position = model_instance.grid.hive

        for _ in range(960):
            model_instance.schedule.step()

        c = 0

        for forager in model_instance.schedule.agents:
            if isinstance(forager, ForagerBeeAgent):
                if forager.status == BeeStatus.LOADING_NECTAR:
                    c += 1

        found_at_day.append(c)

        if c == 500 or it >= 20:
            break

    return found_at_day

def parallel_run(num_processes, num_iterations, params):
    # Create a list of parameter tuples for each iteration
    param_list = [params for _ in range(num_iterations)]

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_bee_model_instance, param_list)

    return results

def __main__(args):
    model = BeeForagingModel(500, 50)
    model.debug(80000)
    print(model.total_energy)




if __name__ == '__main__':
    __main__(sys.argv)

