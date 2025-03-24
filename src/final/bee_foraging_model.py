#packages for modeling
import math
import sys
import warnings
from functools import cached_property
from typing import Tuple

import mesa

#packages for feature implementation
from enum import Enum
import random

#packages for multicore processing
import multiprocessing as mp
import itertools

#packages for data analysis
import csv
import pandas as pd
import numpy as np
from pandas.core.roperator import rand_

### definition of fixed global variables ###
STEPS_PER_DAY = 24 * 3600 # amount of ticks in a full day
MAX_POST_COLLECTION_CLUSTERING_TIME = 4.5 * 3600

RESTING_ENERGY_COST = 6.2 * (1.0/1000) #mJ/s
UNLOADING_NECTAR_ENERGY_COST = 9.3 * (1.0/1000) #mJ/s
FLYING_COST_UNLOADED = 37.0 * (1.0/1000) #mJ/s
FLYING_COST_LOADED = 75 * (1.0/1000) #mJ/s
FLYING_SPEED = 6.944 #m/s
SEARCHING_SPEED = FLYING_SPEED / 3 #m/s
SEARCHING_ANGLE_RANGE = (-120.0, 120.0) #°

MAX_SEARCH_TIME = 960 #s [or ticks]

NECTAR_REWARD = 353.8 #*C J
MORTALITY_RATE = 7.7 * 10 ** (-7)
DAY_SKIPPER_RATE = 20 # %

### definition of model specific parameters ###
MAX_SIGHT = 1
MIN_BORDER_DISTANCE = 3
MAX_SEARCH_RADIUS = 50  #m max radius in which bee looks for the source

### definition of normal distribution variables for random drawing ###
MEAN = 0
STANDARD_DEVIATION = 35

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
class BeeState(Enum):
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
    DAY_SKIPPING = 12,

class ForagingStrategy(Enum):
    PERSISTENT = 1,
    RETICENT = 2,

class BeeForagingModel(mesa.Model):
    """
    Model instance for the Simulation
    """

    def __init__(self, source_distance : int,
                 number_of_starting_bees : int,
                 sucrose_concentration : float = 1.0,
                 sunrise : int = 7 * 3600,
                 sunset : int = 19 * 3600,
                 anticipation_method : int = 1,
                 flower_open : int = 9 * 3600,
                 flower_closed : int = 14 * 3600) -> None:
        """
        Initializes the BeeForagingModel instance

        :param source_distance: distance between source and hive
        :param number_of_starting_bees: number of ForagerBeeAgents at the beginning of the simulation
        :param sucrose_concentration: sucrose concentration of the source
        """
        super().__init__()

        self.sunrise = sunrise  # tick when the sun rises
        self.sunset = sunset  # tick when the sun sets
        self.anticipation_method = anticipation_method
        self.number_of_starting_bees = number_of_starting_bees
        self.sucrose_concentration = sucrose_concentration

        self.total_energy = 0   # this is the final output for our model

        # create a simulated grid
        self.size = (source_distance + 70) * 2
        self.hive = (self.size // 2, self.size // 2)
        self.dance_floor = (self.hive[0] + 1, self.hive[1])

        # create flower (food source) and place it on the grid
        flower_location = generate_random_point(self.hive[0], self.hive[1], source_distance, 0.01)
        flower = FlowerAgent(self, flower_location, 1000, flower_open, flower_closed, sucrose_concentration)
        self.agents.add(flower)

        # add the desired number of Bee Agents to the grid
        # The foragers come with one day of experience already and randomly assigned anticipation times
        for i in range(number_of_starting_bees):
            bee_agent = ForagerBeeAgent(self, 1,
                                        (self.hive[0], self.hive[1]),
                                        time_source_found=random.randint(flower.open_time, flower.close_time))
            bee_agent.targeted_flower = flower
            bee_agent.next_anticipation_time = bee_agent.anticipation(1, self.sunrise, self.sunset)
            C = self.sucrose_concentration
            bee_agent.last_collection_time = random.randint(flower.close_time - (random.randint(0,
                                                                                                int(2 * flower.flight_duration) + int(random.uniform(39 * (C ** 2) + 114.1 * C - 64.25, 159 * (C ** 2) - 140 * C + 166)))), flower.close_time)
            bee_agent.collection_times.append(bee_agent.last_collection_time)
            self.agents.add(bee_agent)

        # assign foraging strategies to the newly created foragers (PERSISTENT/RETICENT)
        self.update_bee_foraging_strategies()

    def debug(self, steps : int) -> None:
        for _ in range(steps):
            self.step()

            if self.steps > 117000:

                for bee in self.agents:
                    if not isinstance(bee, ForagerBeeAgent):
                        continue

    def run(self, steps : int) -> None:
        """
        runs the model for an arbitrary number of steps

        :param steps: number of steps to run
        """

        for _ in range(steps):
            self.step()

    def out_of_bounds(self, pos : Tuple[float, float]) -> bool:
        """
        Determines if a given position is still on the virtual grid

        :param pos: position to be tested
        :return:
        """
        x, y = pos
        return x < 0 or x >= self.size or y < 0 or y >= self.size

    def get_bees(self) -> list:
        """
        Returns all bee agents of the model

        :return: list of bee agents
        """
        return [bee for bee in self.agents if isinstance(bee, ForagerBeeAgent)]

    def get_bees_on_dance_floor(self) -> list:
        """
        Returns all bee agents of the model that are currently on the dance floor

        :return: list of bee agents
        """
        return [bee for bee in self.get_bees() if bee.accurate_position == self.dance_floor]

    @cached_property
    def flowers(self) -> list:
        """
        Returns all flower agents of the model

        :return: list of flower agents
        """
        return [flower for flower in self.agents if isinstance(flower, FlowerAgent)]

    @property
    def current_day(self) -> int:
        """
        Returns the current day of the simulation

        The simulation starts at day 0
        :return: current day
        """
        return self.steps // STEPS_PER_DAY

    def update_bee_foraging_strategies(self) -> None:
        """
        Reassigns foraging strategies to all bees based on their current experience
        """
        self.reassign_foraging_strategies(1, 40)
        self.reassign_foraging_strategies(2, 60)
        self.reassign_foraging_strategies(3, 80)
        self.reassign_foraging_strategies(4, 90)
        self.reassign_foraging_strategies_eq_and_higher(5, 95)

    def reassign_foraging_strategies(self, days_of_experience : int, persistent_percentage : int | float) -> None:
        """
        Reassigns foraging strategies to a group of bees with a specific number of days of experience

        :param days_of_experience: number of days of experience
        :param persistent_percentage: percentage of persistent foragers
        """

        bees = [bee for bee in self.agents if isinstance(bee, ForagerBeeAgent) and
                bee.days_of_experience == days_of_experience and
                bee.state != BeeState.DAY_SKIPPING]

        bee_groups = split_agents_by_percentage(bees, persistent_percentage)

        for bee in bee_groups[0]:
            bee.foraging_strategy = ForagingStrategy.PERSISTENT

        for bee in bee_groups[1]:
            bee.foraging_strategy = ForagingStrategy.RETICENT

    def reassign_foraging_strategies_eq_and_higher(self, days_of_experience : int, persistent_percentage : int | float) -> None:
        """
        Reassigns foraging strategies to a group of bees with a specific number of days of experience or more experience

        :param days_of_experience: min number of days of experience
        :param persistent_percentage: percentage of persistent foragers
        """
        bees = [bee for bee in self.agents if isinstance(bee, ForagerBeeAgent) and
                bee.days_of_experience >= days_of_experience and
                bee.state != BeeState.DAY_SKIPPING]

        bee_groups = split_agents_by_percentage(bees, persistent_percentage)

        for bee in bee_groups[0]:
            bee.foraging_strategy = ForagingStrategy.PERSISTENT

        for bee in bee_groups[1]:
            bee.foraging_strategy = ForagingStrategy.RETICENT

    def reassign_day_skippers(self, percentage : int | float =20) -> None:
        """
        Chooses a random sample of bee agents and sets their state to DAY_SKIPPING

        :param percentage: percentage of day skippers
        """

        # set all current day skippers to RESTING
        for bee in self.agents:
            if not isinstance(bee, ForagerBeeAgent):
                continue

            bee.state = BeeState.RESTING if bee.state == BeeState.DAY_SKIPPING else bee.state

        bee_groups = split_agents_by_percentage(self.get_bees(), percentage)
        for bee in bee_groups[0]:
            bee.state = BeeState.DAY_SKIPPING

    def add_new_foragers(self, number_of_new_foragers : int) -> None:
        """
        Adds new inexperienced foragers to the model

        :param number_of_new_foragers: number of foragers to be added to the model
        """

        for _ in range(number_of_new_foragers):
            new_bee_agent = ForagerBeeAgent(self, 0, (self.hive[0], self.hive[1]))
            new_bee_agent.next_anticipation_time = self.sunrise + self.current_day * STEPS_PER_DAY
            new_bee_agent.foraging_strategy = ForagingStrategy.RETICENT
            self.agents.add(new_bee_agent)

    def kill_agents(self, percentage: int | float =20) -> None:
        """
        Removes bee agents from the model

        :param percentage: percentage of foragers to be removed
        """

        # Get all bee agents
        bee_agents = [agent for agent in self.agents if isinstance(agent, ForagerBeeAgent)]

        # Calculate how many to kill
        num_to_kill = int(len(bee_agents) * percentage / 100.0)

        # Select which bees to kill randomly
        bees_to_kill = random.sample(bee_agents, num_to_kill)

        # Remove these bees from the model
        for bee in bees_to_kill:
            self.agents.remove(bee)

    def today(self, time : int | float) -> int:
        """
        Returns what time step is the same time on the current day as a provided timestep on an arbitrary day

        :param time: time step to be converted into today's time step
        :return: integer determining today's equivalent step
        """
        if not isinstance(time, int):
            time = int(time)

        return time % STEPS_PER_DAY + self.current_day * STEPS_PER_DAY

    def daily_update(self) -> None:
        """
        Updates variables of the model usually at the beginning of a new day of the simulation:
            ==> day skippers are reassigned
            ==> foraging strategies are reassigned
            ==> new foragers are added to the model
        """

        self.reassign_day_skippers(DAY_SKIPPER_RATE)
        self.update_bee_foraging_strategies()
        self.add_new_foragers(self.number_of_starting_bees)

    def step(self) -> None:
        """
        Performs a single step of the simulation
        """

        self.agents.do("step")

        # every day the foraging strategies are reassigned, new foragers are added, and new day skippers are determined
        # on the first day of the simulation we already manually created all the agents so we skip the update on the first day
        # on step one of a new day the Agents update their variables, so the model updates its variables on step 2 of each day
        #  to ensure the agents' variables are up to date

        if not self.steps == 2 and self.steps % STEPS_PER_DAY == 2:
            self.daily_update()

class FlowerAgent(mesa.Agent):
    """
    Models a flower/food source in the model
    """
    def __init__(self, bee_model : BeeForagingModel,
                 location : Tuple[int, int] | Tuple[float, float],
                 sucrose_stock : int | float,
                 open_time : int,
                 close_time : int,
                 sucrose_concentration : float,
                 visibility_radius : float = 1.0,
                 bloom_state : Bloom =Bloom.CLOSED,
                 color : Color =random.choice(list(Color))) -> None:
        """
        Initializes a FlowerAgent object

        :param bee_model: model instance the flower is supposed to be added to
        :param location: location of the flower
        :param sucrose_stock: available food stock
        :param open_time: time step the flower opens
        :param close_time: time step the flower closes
        :param sucrose_concentration: sucrose concentration of the flower
        :param visibility_radius: distance from which ForagerBeeAgents can see the flower
        :param bloom_state: current bloom state of the flower; determines if food is accessible
        :param color: bloom color
        """

        if not isinstance(bee_model, BeeForagingModel):
            raise TypeError("bee_model must be of type BeeForagingModel")

        super().__init__(bee_model)

        self.sucrose_stock = sucrose_stock   # available sucrose in flower agent
        self.open_time = int(open_time)   # tick/step when the flower opens ==> bee can access the flower
        self.close_time = int(close_time)   # tick/step when the flower closes ==> no bee can access the flower
        self.location = location   # location of the flower
        self.sucrose_concentration = sucrose_concentration
        self.visibility_radius = visibility_radius
        self.bloom_state = bloom_state
        self.color = color

    @cached_property
    def flight_duration(self) -> float:
        return get_distance(self.location, self.model.hive) / FLYING_SPEED

    @cached_property
    def distance_from_hive(self):
        return get_distance(self.location, self.model.hive)

    @cached_property
    def value(self):
        C = self.sucrose_concentration
        D = self.distance_from_hive
        return (353.6 * C - 0.015 * D - 2.64) / (0.2 * C + 0.015 * D + 2.64)

    def step(self) -> None:
        """
        Performs a single step for the agent in the model
        """

        if self.model.steps % STEPS_PER_DAY == self.open_time:
            self.bloom_state = Bloom.OPEN

        if self.model.steps % STEPS_PER_DAY == self.close_time:
            self.bloom_state = Bloom.CLOSED

    def __str__(self):
        return (
            f"Flower at {self.location}\n"
            f"Bloom State: {'OPEN' if self.bloom_state == Bloom.OPEN else 'CLOSED'}\n"
            f"Color: {self.color}\n"
            f"Opening Hours: {self.open_time} - {self.close_time}\n"
            f"Distance from Hive: {self.distance_from_hive:.1f}m\n"
            f"Flight Duration: {self.flight_duration:.1f} ticks\n"
            f"Nectar Stock: {self.sucrose_stock:.1f}μl\n"
            f"Sucrose Concentration: {self.sucrose_concentration:.0f}%\n"
            f"Visibility Radius: {self.visibility_radius}m\n"
            f"Value for Bees: {self.value:.2f}"
        )

class ForagerBeeAgent(mesa.Agent):
    """
    Models a bee forager in the model
    """
    def __init__(self, bee_model : BeeForagingModel,
                 days_of_experience : int,
                 start_pos : Tuple[int, int] | Tuple[float, float],
                 time_source_found : int =-1) -> None:

        """
        Initializes a ForagerBeeAgent object

        :param bee_model: model instance the bee is added to
        :param days_of_experience: number of days the bee has collected food at the flower
        :param start_pos: starting position of the bee
        :param time_source_found: first time of food collection at the targeted_flower on the previous day / current day
        """
        super().__init__(bee_model)

        self.days_of_experience = days_of_experience   # number of days the bee has collected food from a source
        self.accurate_position = start_pos   # current position of the bee
        self.state = BeeState.RESTING   # state of the bee
        self._remaining_time_in_state = 0   # number of steps the bee will remain in the current state (is not necessarily assigned in each state)
        self.time_source_found = time_source_found   # first time of food collection at the targeted_flower on the previous day / current day
        self.last_collection_time = None   # last time the bee found the flover open on the current or previous day
        self.next_anticipation_time = None   # next time the bee anticipates the flower to be open and clusters on the dance floor
        self.foraging_strategy = None   # strategy of the bee (PERSISTENT OR RETICENT)
        self.next_reconnaissance_time = None   # determines the next reconnaissance time step for PERSISTENT foragers
        self.loaded = False   # bee has currently food loaded
        self.collection_times = []   # list of time steps when the bee collected food
        self._mortality_probability = MORTALITY_RATE
        self.targeted_flower = None
        self._last_angle = 0.0
        self._search_area_center = None
        self.currently_watched_bee = None
        self.homing_motivation = 0


    def anticipation(self, method : int,
                     sunrise : int,
                     sunset : int) -> int:
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
            anticipation = int(time_source_found - distance/FLYING_SPEED)
            return int(self.model.today(anticipation))

        elif method == 2:
            anticipation = (time_source_found - 3600 * 4 * (time_source_found - sunrise) / (sunset - sunrise)) - distance/FLYING_SPEED
            return self.model.today(int(anticipation))

        else:
            return -1

    def search(self, flower : FlowerAgent, search_area_center : Tuple[float, float]) -> None:
        """
        Models the bee search behaviour
            => If the bee is closer than MAX_SIGHT  to the source it or was in this range sometimes between the current and last time step it directly flies to the flower
            => If the bee

        :param flower: the position the bee tries to find
        :param search_area_center: a tuple that describes the area in which the bee tries to find the destination
        """
        self._search_area_center = search_area_center

        if get_distance(self.accurate_position, flower.location) <= flower.visibility_radius or self.last_move_crossed_flower_radius(self._last_angle, SEARCHING_SPEED, flower):
            self.move_bee_towards_point(flower.location, SEARCHING_SPEED)

        elif get_distance(self.accurate_position, self._search_area_center) >= MAX_SEARCH_RADIUS:
            angle = get_angle(self.accurate_position, self._search_area_center)
            angle = random_deviate_angle_equally(angle, SEARCHING_ANGLE_RANGE[0], SEARCHING_ANGLE_RANGE[1])
            self.move_bee_with_angle(angle, SEARCHING_SPEED)

        else:
            angle = random_deviate_angle(self._last_angle, MEAN, STANDARD_DEVIATION, SEARCHING_ANGLE_RANGE[0], SEARCHING_ANGLE_RANGE[1])
            self.move_bee_with_angle(angle, SEARCHING_SPEED)

        self.homing_motivation += 1

    def move_bee_towards_point(self, destination : Tuple[float, float], speed : float) -> None:

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
            self._last_angle = angle

        else:
            new_pos = (self.accurate_position[0] + speed * math.cos(angle), self.accurate_position[1] + speed * math.sin(angle))
            if not self.model.out_of_bounds(new_pos):
                self.accurate_position = new_pos
                self._last_angle = angle

    def move_bee_with_angle(self, angle : float, speed : float) -> None:
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
            self._last_angle = angle

    def last_move_crossed_flower_radius(self, last_angle : float, speed : float, flower : FlowerAgent) -> bool:
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

    @property
    def late_reconnaissance_probability(self) -> float:
        """
        Probability of late reconnaissance flight for persistent forager bees

        :return: probability for bee too fly out
        """

        if len(self.collection_times) < 1:
            return 0.0

        t = self.model.steps % STEPS_PER_DAY - self.collection_times[-1]
        return -2 * (10 ** (-16)) * (t ** 3) + 5 * (10 ** (-12)) * (t ** 2) - 5 * (10 ** (-8)) * t + 0.0002

    def daily_variable_reassignment(self) -> None:
        """
        Reassign all bee agent specific variables for the day on time step 1
        """

        if not isinstance(self.model, BeeForagingModel):
            raise TypeError("bee_model must be of type BeeForagingModel")

        # reassign the next anticipation time when the bee is going to cluster
        self.next_anticipation_time = self.anticipation(self.model.anticipation_method, self.model.sunrise, self.model.sunset) if self.targeted_flower is not None else self.model.today(self.model.sunrise)

        # if the bee has been collecting nectar on the previous day its number of days of experience is increased
        if (self.last_collection_time is not None and
                self.state != BeeState.DAY_SKIPPING and
                get_day_of_step(self.last_collection_time) == self.model.current_day - 1):

           self.days_of_experience += 1

        # for persistent foragers the next time they will do a reconnaissance flight (if not employed before) is assigned as well
        if self.foraging_strategy == ForagingStrategy.PERSISTENT:
            self.next_reconnaissance_time = random.randint(self.next_anticipation_time, (self.time_source_found % STEPS_PER_DAY) + (self.model.current_day) * STEPS_PER_DAY)

        else:
            self.next_reconnaissance_time = None

    def update_status(self) -> None:
        """
        Updates the current status of the bee after each time step
        """
        if not isinstance(self.model, BeeForagingModel):
            raise ValueError("The bee foraging model must be a BeeForagingModel")

        # on the last step of the day, the last collection time is updated
        if self.model.steps % STEPS_PER_DAY == STEPS_PER_DAY - 1:
            if len(self.collection_times)  > 0:
                self.last_collection_time = self.collection_times[-1]

        # on step 1 of each day the variables of the agents are updated
        if self.model.steps % STEPS_PER_DAY == 1:
            self.daily_variable_reassignment()

        if self.state == BeeState.RESTING:
            # if anticipation time comes up the state is set to CLUSTERING
            if self.model.steps % STEPS_PER_DAY == self.next_anticipation_time % STEPS_PER_DAY:
                self.state = BeeState.CLUSTERING
                self.accurate_position = self.model.dance_floor

        # bee is CLUSTERING  ==> can see bees on the dance floor and can be seen by others on the dance floor
        elif self.state == BeeState.CLUSTERING:

            # at sunset the bee enters resting state if she has not yet
            if self.model.steps % STEPS_PER_DAY >= self.model.sunset % STEPS_PER_DAY:
                self.state = BeeState.RESTING
                self.accurate_position = self.model.hive

            # after last collection time + max post collection clustering time the bee enters resting state if she has not yet
            elif self.last_collection_time is not None and self.last_collection_time % STEPS_PER_DAY + MAX_POST_COLLECTION_CLUSTERING_TIME < self.model.steps % STEPS_PER_DAY:
                self.state = BeeState.RESTING
                self.accurate_position = self.model.hive

            # if the bee is watching another bee that is dancing, and she is inexperienced or is targeting the same flower she will watch the waggle dance for several durations
            elif self.currently_watched_bee is not None and (self.targeted_flower is self.currently_watched_bee.targeted_flower or self.targeted_flower is None):
                self.state = BeeState.WATCHING_WAGGLE_DANCE
                self.targeted_flower = self.currently_watched_bee.targeted_flower if self.targeted_flower is None else self.targeted_flower
                self._remaining_time_in_state = int(((0.0013 * self.targeted_flower.distance_from_hive) + 1.86) * random.uniform(4.9, 12.9))

            #
            elif self.foraging_strategy == ForagingStrategy.PERSISTENT and self.next_reconnaissance_time == self.model.steps:
                self.state = BeeState.FLYING_STRAIGHT_TO_FLOWER

            elif self.foraging_strategy == ForagingStrategy.PERSISTENT and self.targeted_flower.close_time < self.model.steps:
                if random.random() < self.late_reconnaissance_probability:
                    self.state = BeeState.FLYING_STRAIGHT_TO_FLOWER


        elif self.state == BeeState.WATCHING_WAGGLE_DANCE:
            if self._remaining_time_in_state > 0:
                self._remaining_time_in_state -= 1

            else:
                if self.days_of_experience >= 0:
                    self.state = BeeState.FLYING_STRAIGHT_TO_FLOWER

                else:
                    self.state = BeeState.FLYING_TO_SEARCH_AREA
                    rand_distance_from_flower = random.randint(1,MAX_SEARCH_RADIUS)
                    self._search_area_center = generate_random_point(self.targeted_flower.location[0], self.targeted_flower.location[1], rand_distance_from_flower)


                self.currently_watched_bee = None


        elif self.state == BeeState.FLYING_STRAIGHT_TO_FLOWER or self.state == BeeState.SEARCHING_ADVERTISED_SOURCE:
            if self.accurate_position == self.targeted_flower.location:
                if self.targeted_flower.bloom_state == Bloom.OPEN:
                    self.state = BeeState.LOADING_NECTAR
                    self.collection_times.append(self.model.steps)

                    C = self.targeted_flower.sucrose_concentration
                    self._remaining_time_in_state = int(random.uniform(12.44 * C + 20.09, 24.22 * C + 32.07))
                    self.time_source_found = self.model.steps if self.time_source_found // STEPS_PER_DAY != self.model.current_day else self.time_source_found

                else:
                    self.state = BeeState.RETURNING

                    if self.foraging_strategy == ForagingStrategy.PERSISTENT and self.next_reconnaissance_time is not None:
                        self.next_reconnaissance_time = int((self.next_reconnaissance_time + self.model.today(self.time_source_found)) / 2) if abs(self.next_reconnaissance_time - self.model.today(self.time_source_found)) > 120 + 2 * self.targeted_flower.flight_duration else self.model.steps + 120 + 2 * self.targeted_flower.flight_duration

            elif self.homing_motivation > MAX_SEARCH_TIME:
                self.state = BeeState.RETURNING
                self.homing_motivation = 0


        elif self.state == BeeState.FLYING_TO_SEARCH_AREA:
            if self.accurate_position == self._search_area_center:
                self.state = BeeState.SEARCHING_ADVERTISED_SOURCE


        elif self.state == BeeState.LOADING_NECTAR:
            if self._remaining_time_in_state > 0:
                self._remaining_time_in_state -= 1

            else:
                self.loaded = True
                self.state = BeeState.RETURNING


        elif self.state == BeeState.RETURNING:
            if self.accurate_position == self.model.dance_floor:
                if self.loaded:
                    self.state = BeeState.UNLOADING_NECTAR
                    C = self.targeted_flower.sucrose_concentration
                    self._remaining_time_in_state = int(random.uniform(39 * (C ** 2) + 114.1 * C - 64.25, 159 * (C ** 2) - 140 * C + 166))

                else:
                    self.state = BeeState.CLUSTERING
                    self.accurate_position = self.model.dance_floor

        elif self.state == BeeState.UNLOADING_NECTAR:
            if self._remaining_time_in_state > 0:
                self._remaining_time_in_state -= 1

            else:
                self.model.total_energy += NECTAR_REWARD * self.targeted_flower.sucrose_concentration
                self.loaded = False
                self.state = BeeState.DANCING
                self._remaining_time_in_state = 0.1713 * self.targeted_flower.value


        elif self.state == BeeState.DANCING:
            if self._remaining_time_in_state > 0:
                self._remaining_time_in_state -= 1

            else:
                self.state = BeeState.PREPARING_TO_FLY_OUT
                self._remaining_time_in_state = random.randint(16, 51)


        elif self.state == BeeState.PREPARING_TO_FLY_OUT:
            if self._remaining_time_in_state > 0:
                self._remaining_time_in_state -= 1

            else:
                self.state = BeeState.FLYING_STRAIGHT_TO_FLOWER

        else:
            self.state = self.state

    def step(self) -> None:
        """
        Step function of the bee
        """
        if not isinstance(self.model, BeeForagingModel):
            raise AttributeError("BeeForagingModel is not an instance of BeeForagingModel")

        if random.random() < self._mortality_probability:
            self.model.agents.remove(self)

        self.update_status()

        if (self.state == BeeState.RESTING or
                self.state == BeeState.WATCHING_WAGGLE_DANCE or
                self.state == BeeState.UNLOADING_NECTAR or
                self.state == BeeState.LOADING_NECTAR or
                self.state == BeeState.DANCING or
                self.state == BeeState.PREPARING_TO_FLY_OUT or
                self.state == BeeState.DAY_SKIPPING):

            self.model.total_energy -= RESTING_ENERGY_COST

        elif self.state == BeeState.CLUSTERING:
            self.model.total_energy -= RESTING_ENERGY_COST
            seen_bees = split_agents_by_percentage(self.model.get_bees_on_dance_floor(), 4)[0]


            for bee in seen_bees:
                if bee.state == BeeState.DANCING:
                    if self.targeted_flower is None or self.targeted_flower == bee.targeted_flower:
                        self.targeted_flower = bee.targeted_flower
                        self.currently_watched_bee = bee
                        break


        elif self.state == BeeState.FLYING_STRAIGHT_TO_FLOWER:
            self.model.total_energy -= FLYING_COST_UNLOADED

            if get_distance(self.accurate_position, self.targeted_flower.location) <= FLYING_SPEED:
                self.accurate_position = self.targeted_flower.location

            else:
                self.move_bee_towards_point(self.targeted_flower.location, FLYING_SPEED)


        elif self.state == BeeState.FLYING_TO_SEARCH_AREA:
            self.model.total_energy -= FLYING_COST_UNLOADED
            if get_distance(self.accurate_position, self._search_area_center) <= FLYING_SPEED:
                self.accurate_position = self._search_area_center


        elif self.state == BeeState.SEARCHING_ADVERTISED_SOURCE:
            self.search(self.targeted_flower.location, self._search_area_center)


        elif self.state == BeeState.RETURNING:
            if self.loaded:
                self.model.total_energy -= FLYING_COST_LOADED

            else:
                self.model.total_energy -= FLYING_COST_UNLOADED


            if get_distance(self.accurate_position, self.model.dance_floor) <= FLYING_SPEED:
                self.accurate_position = self.model.dance_floor

            else:
                self.move_bee_towards_point(self.model.dance_floor, FLYING_SPEED)

    def __str__(self) -> str:
        return (
            f"Bee status: {self.state}\n"
            f"Experience: {self.days_of_experience} days\n"
            f"Position: {self.accurate_position}\n"
            f"Loaded: {'Yes' if self.loaded else 'No'}\n"
            f"Time in state remaining: {self._remaining_time_in_state}\n"
            f"Time source found: {self.time_source_found}\n"
            f"Last collection time: {self.last_collection_time}\n"
            f"Next anticipation time: {self.next_anticipation_time}\n"
            f"Foraging strategy: {self.foraging_strategy}\n"
            f"Next reconnaissance time: {self.next_reconnaissance_time}\n"
            f"Collection times: {self.collection_times}\n"
            f"Mortality probability: {self._mortality_probability:.4f}\n"
            f"Targeted flower: {self.targeted_flower}\n"
            f"Last angle: {self._last_angle:.2f}\n"
            f"Search area center: {self._search_area_center}\n"
            f"Currently watched bee: {self.currently_watched_bee}\n"
            f"Homing motivation: {self.homing_motivation}"
        )

def generate_random_point(origin_x : int | float,
                          origin_y: int | float,
                          target_distance : int | float,
                          tolerance: float =0.1,
                          max_attempts : int=10000) -> Tuple[int, int] | None:
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

def get_next_point(current_x : int | float,
                   current_y : int | float,
                   angle : float, distance : float) -> Tuple[float, float] | None:
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

def get_distance(pos1 : Tuple[float, float], pos2 : Tuple[float, float]) -> float:
    """
        Use euclidian distance to calculate the distance between two points
    """
    return round(math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2), 2)

def get_angle(starting_point: Tuple[float, float], destination_point : Tuple[float, float]) -> float:
    """
        the angle in a triangle is calculated by arctan(y/x)

        We use this principle to calculate the angle/direction in which an object would have to
        be moved to reach a given point
    """

    dx = destination_point[0] - starting_point[0]
    dy = destination_point[1] - starting_point[1]

    angle = math.atan2(dy, dx)

    return normalize_angle(angle)

def random_deviate_angle(current_angle : float,
                         mean : int | float,
                         standard_deviation : int | float,
                         min_value: int | float,
                         max_value: int | float,
                         radians : bool =False) -> float:
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

def random_deviate_angle_equally(angle : float,
                                 min_value : int | float,
                                 max_value : int | float,
                                 radians : bool=False):

    deviation = random.uniform(min_value, max_value)

    if radians:
        angle += deviation

    else:
        angle += math.radians(deviation)

    return normalize_angle(angle)

def normalize_angle(current_angle : float) -> float:
    angle = current_angle % (2 * math.pi)

    if angle < 0:
        angle += 2 * math.pi

    return angle

def radians_to_degrees(radians : int | float) -> float:
    return radians * (180 / math.pi)

def circle_line_intersect(p1 : Tuple[float, float], p2 : Tuple[float, float], circle_center : Tuple[float, float], radius : float) -> bool:
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

def draw_normal_distributed_value(mean : int | float,
                                  standard_deviation : int | float,
                                  min_value : int | float,
                                  max_value : int | float) -> float:
    while True:
        value = random.normalvariate(mean, standard_deviation)
        if min_value <= value <= max_value:
            return value

def get_day_of_step(step : int) -> int:
    return step // STEPS_PER_DAY

def split_agents_by_percentage(agents : list, first_percentage : int | float =30):
    """

    :param agents:
    :param first_percentage:
    :return:
    """

    if not agents:
        return [], []

    # Ensure percentage is within bounds
    first_percentage = max(0, min(100, first_percentage))

    # Calculate how many agents go in the first group
    num_in_first = int(len(agents) * (first_percentage / 100))
    if num_in_first == 0:
        num_in_first = 1

    # Create a copy and shuffle it
    shuffled_agents = agents.copy()
    random.shuffle(shuffled_agents)

    # Split the shuffled agents
    first_group = shuffled_agents[:num_in_first]
    second_group = shuffled_agents[num_in_first:]

    return first_group, second_group

def run_model_instance(time_steps, **params):
    model = BeeForagingModel(**params)

    model.run(time_steps)
    return model.total_energy

def run_single_model_instance(args):
    """
    Wrapper function that unpacks arguments for run_model_instance

    :param args: A tuple containing (time_steps, specific_params)
    :return: Result of the model run
    """
    time_steps, specific_params = args
    return run_model_instance(time_steps, **specific_params)


def parallel_run(num_cores, num_runs_per_combination, time_steps, params):
    """
    Run parallel simulations

    :param num_cores: Number of CPU cores to use
    :param num_runs_per_combination: Number of times to run each parameter combination
    :param time_steps: Number of simulation steps
    :param params: List of parameter dictionaries
    :return: Aggregated results from all runs
    """
    # Expand params to include multiple runs
    expanded_params = [(time_steps, param) for param in params for _ in range(num_runs_per_combination)]

    # Create a process pool
    with mp.Pool(processes=num_cores) as pool:
        # Use map with the wrapper function
        results = pool.map(run_single_model_instance, expanded_params)

    return results

def main(args):
    number_of_starting_foragers = [10, 20]
    source_distance = [500, 1000]
    sucrose_concentration = [1, 1.5, 2]
    anticipation_method = [1, 2]

    number_of_steps = 3000
    number_of_runs_per_combination = 10

    params = []
    for n, d, c, a in itertools.product(
        number_of_starting_foragers,
        source_distance,
        sucrose_concentration,
        anticipation_method
    ):
        params.append({
            "number_of_starting_bees": n,
            "source_distance": d,
            "sucrose_concentration": c,
            "anticipation_method": a
        })

    print(parallel_run(
        mp.cpu_count() - 2,
        number_of_runs_per_combination,
        number_of_steps,
        params
    ))

if __name__ == "__main__":
    main(sys.argv)