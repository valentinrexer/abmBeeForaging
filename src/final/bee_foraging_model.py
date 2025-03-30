from __future__ import annotations

import os
import sys
import logging
from functools import cached_property

#packages for feature implementation
import mesa

from custom_enum import *
from geometry import *
from const import *

import random
import math

#packages for multicore processing
import multiprocessing as mp
import itertools

#packages for data analysis
import csv

_LOGGER = logging.getLogger(__name__)


class BeeForagingModel(mesa.Model):
    """
    Model instance for the Simulation
    """

    def __init__(self, source_distance : int,
                 number_of_starting_bees : int,
                 sucrose_concentration : float = 1.0,
                 sunrise : int = 7 * STEPS_PER_HOUR,
                 sunset : int = 19 * STEPS_PER_HOUR,
                 anticipation_method : int = 1,
                 flower_open : int = 9 * STEPS_PER_HOUR,
                 flower_closed : int = 14 * STEPS_PER_HOUR,
                 collector_path : str = None,
                 collection_interval : int = None) -> None:
        """
        Initializes the model instance

        :param source_distance: distance from the food source to the hive
        :param number_of_starting_bees: number of starting foragers
        :param sucrose_concentration: sucrose concentration of the source
        :param sunrise: sunrise for each day
        :param sunset: sunset for each day
        :param anticipation_method: anticipation strategy of the foragers for anticipating cluster time
        :param flower_open: first time step when food is available at the flower
        :param flower_closed: time step after which been can no longer collect food at the flower
        :param collector_path: file path to the output csv file
        :param collection_interval: interval between data collection time steps
        """
        super().__init__()

        self.sunrise = sunrise  # tick when the sun rises
        self.sunset = sunset  # tick when the sun sets
        self.anticipation_method = anticipation_method
        self.number_of_starting_bees = number_of_starting_bees
        self.sucrose_concentration = sucrose_concentration
        self._source_distance = source_distance

        self.total_energy = 0   # this is the final output for our model

        # create a simulated grid
        self.size = (source_distance + 70) * 2
        self.hive = (self.size // 2, self.size // 2)
        self.dance_floor = (self.hive[0] + 1, self.hive[1])

        # create flower (food source) and place it on the grid
        flower_location = Calc.generate_random_point(self.hive[0], self.hive[1], source_distance, 0.01)
        flower = FlowerAgent(self, flower_location, 1000, flower_open, flower_closed, sucrose_concentration)
        self.agents.add(flower)

        # add the desired number of Bee Agents to the grid
        # The foragers come with one day of experience already and randomly assigned anticipation times
        for i in range(number_of_starting_bees):
            bee_agent = ForagerBeeAgent(self, 1,
                                        (self.hive[0], self.hive[1]),
                                        time_source_found=random.randint(flower.open_time, flower.open_time + 2 * STEPS_PER_HOUR))
            bee_agent.targeted_flower = flower
            bee_agent.next_anticipation_time = bee_agent.anticipation(1, self.sunrise, self.sunset)
            C = self.sucrose_concentration
            bee_agent.last_collection_time = random.randint(flower.close_time - (random.randint(0,
                                                                                                int(2 * flower.flight_duration) + int(random.uniform(39 * (C ** 2) + 114.1 * C - 64.25, 159 * (C ** 2) - 140 * C + 166)))), flower.close_time)
            bee_agent.collection_times.append(bee_agent.last_collection_time)
            self.agents.add(bee_agent)

        # assign foraging strategies to the newly created foragers (PERSISTENT/RETICENT)
        self.update_bee_foraging_strategies()


        self.collector = (DataCollector(self, collector_path, collection_interval)
                          if collector_path is not None
                             and collection_interval is not None
                          else None)

    def run(self, steps : int) -> None:
        """
        runs the model for an arbitrary number of steps

        :param steps: number of steps to run
        """
        for _ in range(steps):
            self.step()

    def out_of_bounds(self, pos : tuple[float, float]) -> bool:
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

    @property
    def initial_source_distance(self) -> float:
        return self._source_distance

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

        bee_groups = BeeForagingModel.split_agents_by_percentage(bees, persistent_percentage)

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

        bee_groups = BeeForagingModel.split_agents_by_percentage(bees, persistent_percentage)

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

        bee_groups = BeeForagingModel.split_agents_by_percentage(self.get_bees(), percentage)
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

        if self.collector is not None:
            self.collector.check_for_collection_call()

        # every day the foraging strategies are reassigned, new foragers are added, and new day skippers are determined
        # on the first day of the simulation we already manually created all the agents so we skip the update on the first day
        # on step one of a new day the Agents update their variables, so the model updates its variables on step 2 of each day
        #  to ensure the agents' variables are up to date

        if not self.steps == 2 and self.steps % STEPS_PER_DAY == 2:
            self.daily_update()

    @staticmethod
    def split_agents_by_percentage(agents: list, first_percentage: int | float = 30, exclude_agent=None) -> tuple[list, list]:
        """
        Split a set of items (in our case agents) into two subsets randomly

        :param agents: list to be divided
        :param first_percentage: size of the first list in percent
        :param exclude_agent: agents to be excluded
        :return: two subset lists with corresponding size
        """

        if not agents:
            return [], []

        if not 1 <= first_percentage <= 100:
            raise ValueError("First percentage must be between 1 and 100")

        # Calculate how many agents go in the first group
        n_agents = len(agents) if exclude_agent is None else len(agents) - 1
        if n_agents == 0:
            return [], []

        # if the first group of the split does not contain an item the number is increased so at least one bee is seen
        num_in_first = int(n_agents * (first_percentage / 100))
        if num_in_first == 0:
            num_in_first = 1

        # Create a copy and shuffle it
        shuffled_agents = agents.copy()

        # remove agent to be excluded
        if exclude_agent is not None and exclude_agent in shuffled_agents:
            shuffled_agents.remove(exclude_agent)

        random.shuffle(shuffled_agents)

        # Split the shuffled agents
        first_group = shuffled_agents[:num_in_first]
        second_group = shuffled_agents[num_in_first:]

        return first_group, second_group

    @staticmethod
    def get_day_of_step(step: int) -> int:
        """
        Returns the day of a given time step
        :param step: step to derive the day
        :return: day on which the step happens
        """
        return step // STEPS_PER_DAY

class FlowerAgent(mesa.Agent):
    """
    Models a flower/food source in the model
    """

    def __init__(self, bee_model : BeeForagingModel,
                 location : tuple[int, int] | tuple[float, float],
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
        self.sucrose_concentration = sucrose_concentration   # source concentration in the flowers nectar
        self.visibility_radius = visibility_radius  # radius from which the flower can be spotted by a bee agent
        self.bloom_state = bloom_state  # state of the bloom
        self.color = color

    @cached_property
    def flight_duration(self) -> float:
        """
        time a one way flight from the flower to the hive takes

        :return: flight duration
        """
        return Calc.get_distance(self.location, self.model.hive) / FLYING_SPEED

    @cached_property
    def distance_from_hive(self) -> float:
        """
        distance from flower to hive

        :return: distance
        """
        return Calc.get_distance(self.location, self.model.hive)

    @cached_property
    def value(self) -> float:
        """
        Returns the value of a source

        :return: value of the source as floating point number
        """

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

    def __str__(self) -> str:
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
                 start_pos : tuple[int, int] | tuple[float, float],
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
        self.targeted_flower = None  # flower object the bee is targeting at the moment
        self._last_angle = 0.0   # last direction of movement of the bee
        self._search_area_center = None   # point at which the bee begins her search for the source / center of the search radius
        self.currently_watched_bee = None   # bee object watched during a waggle dance
        self.homing_motivation = 0   # time passed while bee is searching the source


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
        distance = Calc.get_distance(self.targeted_flower.location, self.model.hive)

        if method == 1:
            anticipation = int(time_source_found - distance/FLYING_SPEED)
            return int(self.model.today(anticipation))

        elif method == 2:
            anticipation = (time_source_found - STEPS_PER_HOUR * 4 * (time_source_found - sunrise) / (sunset - sunrise)) - distance/FLYING_SPEED
            return self.model.today(int(anticipation))

        else:
            return -1

    def search(self, flower : FlowerAgent, search_area_center : tuple[float, float]) -> None:
        """
        Models the bee search behaviour
            => If the bee is closer than MAX_SIGHT  to the source it or was in this range sometimes between the current and last time step it directly flies to the flower
            => If the bee

        :param flower: the position the bee tries to find
        :param search_area_center: a tuple that describes the area in which the bee tries to find the destination
        """
        self._search_area_center = search_area_center

        # if bee agent is within the visibility radius of the flower or crossed the radius with its last move
        # the bee is moved to the flower location
        if (Calc.get_distance(self.accurate_position, flower.location) <= flower.visibility_radius or
                self.last_move_crossed_flower_radius(self._last_angle, SEARCHING_SPEED, flower)):
            self.move_bee_towards_point(flower.location, SEARCHING_SPEED)

        # if the bee left the search radius during the last step it turns around and flies towards the search area center with
        # slight deviation
        elif Calc.get_distance(self.accurate_position, self._search_area_center) >= MAX_SEARCH_RADIUS:
            angle = Calc.get_angle(self.accurate_position, self._search_area_center)
            angle = Calc.random_deviate_angle_equally(angle, SEARCHING_ANGLE_RANGE[0], SEARCHING_ANGLE_RANGE[1])
            self.move_bee_with_angle(angle, SEARCHING_SPEED)

        # if the bee didn't find the flower during the last step its flying direction is deviated
        else:
            angle = Calc.random_deviate_angle(self._last_angle, MEAN, STANDARD_DEVIATION, SEARCHING_ANGLE_RANGE[0], SEARCHING_ANGLE_RANGE[1])
            self.move_bee_with_angle(angle, SEARCHING_SPEED)

        self.homing_motivation += 1

    def move_bee_towards_point(self, destination : tuple[float, float], speed : float) -> None:

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

        angle = Calc.get_angle(self.accurate_position, destination)
        current_distance = Calc.get_distance(self.accurate_position, destination)

        if current_distance <= speed:
            self.accurate_position = destination
            self._last_angle = angle

        else:
            new_pos = (self.accurate_position[0] + speed * math.cos(angle),
                       self.accurate_position[1] + speed * math.sin(angle))
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

        new_pos = (self.accurate_position[0] + speed * math.cos(angle),
                   self.accurate_position[1] + speed * math.sin(angle))

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

        return Calc.circle_line_intersect((last_x, last_y),
                                          (curr_x, curr_y),
                                          flower.location,
                                          flower.visibility_radius)

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
        self.next_anticipation_time = self.anticipation(self.model.anticipation_method,
                                                        self.model.sunrise,
                                                        self.model.sunset) if self.targeted_flower is not None else self.model.today(self.model.sunrise)

        # if the bee has been collecting nectar on the previous day its number of days of experience is increased
        if (self.last_collection_time is not None and
                self.state != BeeState.DAY_SKIPPING and
                BeeForagingModel.get_day_of_step(self.last_collection_time) == self.model.current_day - 1):

           self.days_of_experience += 1

        # for persistent foragers the next time they will do a reconnaissance flight (if not employed before) is assigned as well
        if self.foraging_strategy == ForagingStrategy.PERSISTENT:
            self.next_reconnaissance_time = random.randint(self.next_anticipation_time,
                                                           (self.time_source_found % STEPS_PER_DAY) + self.model.current_day * STEPS_PER_DAY)

        else:
            self.next_reconnaissance_time = None

    def update_state_resting(self):
        """
        Update the bee state when currently in state RESTING
        """
        if self.model.steps % STEPS_PER_DAY == self.next_anticipation_time % STEPS_PER_DAY:
            self.state = BeeState.CLUSTERING
            self.accurate_position = self.model.dance_floor

    def update_state_clustering(self):
        """
        Update the bee state when currently in state CLUSTERING
        """

        # at sunset the bee enters resting state if she has not yet
        if self.model.steps % STEPS_PER_DAY >= self.model.sunset % STEPS_PER_DAY:
            self.state = BeeState.RESTING
            self.accurate_position = self.model.hive

        # after last collection time + max post collection clustering time the bee enters resting state if she has not yet
        elif self.last_collection_time is not None and self.last_collection_time % STEPS_PER_DAY + MAX_POST_COLLECTION_CLUSTERING_TIME < self.model.steps % STEPS_PER_DAY:
            self.state = BeeState.RESTING
            self.accurate_position = self.model.hive

        # if the bee is watching another bee that is dancing, and she is inexperienced or is targeting the same flower she will watch the waggle dance for several durations
        elif self.currently_watched_bee is not None and (
                self.targeted_flower is self.currently_watched_bee.targeted_flower or self.targeted_flower is None):
            self.state = BeeState.WATCHING_WAGGLE_DANCE
            self.targeted_flower = self.currently_watched_bee.targeted_flower if self.targeted_flower is None else self.targeted_flower
            self._remaining_time_in_state = int(
                ((0.0013 * self.targeted_flower.distance_from_hive) + 1.86) * random.uniform(4.9, 12.9))

        # if the reconnaissance time of a persistent forager comes up the bee flies to the flower
        elif self.foraging_strategy == ForagingStrategy.PERSISTENT and self.next_reconnaissance_time == self.model.steps:
            self.state = BeeState.FLYING_STRAIGHT_TO_FLOWER

        # after the source closes the persistent forager flies out with a small probability
        elif self.foraging_strategy == ForagingStrategy.PERSISTENT and self.targeted_flower.close_time < self.model.steps:
            if random.random() < self.late_reconnaissance_probability:
                self.state = BeeState.FLYING_STRAIGHT_TO_FLOWER

    def update_state_watching_waggle_dance(self):
        """
        Update the bee state when currently in state WATCHING_WAGGLE_DANCE
        """
        if self._remaining_time_in_state > 0:
            self._remaining_time_in_state -= 1

        else:
            if self.days_of_experience > 0:
                self.state = BeeState.FLYING_STRAIGHT_TO_FLOWER

            else:
                self.state = BeeState.FLYING_TO_SEARCH_AREA
                rand_distance_from_flower = random.randint(1, MAX_SEARCH_RADIUS)
                self._search_area_center = Calc.generate_random_point(self.targeted_flower.location[0],
                                                                      self.targeted_flower.location[1],
                                                                      rand_distance_from_flower)

            self.currently_watched_bee = None

    def update_state_flying_to_source_or_searching(self):
        """
        Update the bee state when currently in state FLYING_STRAIGHT_TO_FLOWER or SEARCHING_ADVERTISED_SOURCE
        """
        if self.accurate_position == self.targeted_flower.location:

            # if the forager reached the flower and the bloom is open it starts loading nectar
            if self.targeted_flower.bloom_state == Bloom.OPEN:
                self.state = BeeState.LOADING_NECTAR
                self.collection_times.append(self.model.steps)

                C = self.targeted_flower.sucrose_concentration
                self._remaining_time_in_state = int(random.uniform(12.44 * C + 20.09, 24.22 * C + 32.07))

                # if the forager finds the flower open for the first time on the current day it updates its time_source_found variable
                self.time_source_found = self.model.steps if self.time_source_found // STEPS_PER_DAY != self.model.current_day else self.time_source_found

            # if the source is still or already closed the forager returns to the hive
            else:
                self.state = BeeState.RETURNING

                if self.foraging_strategy == ForagingStrategy.PERSISTENT and self.next_reconnaissance_time is not None:

                    # the next reconnaissance time is in half the time of the interval between the last reconnaissance flight and the current time
                    # if that interval would be smaller than 120s the interval is just set to 120s
                    # since the reconnaissance time is updated at the source the flight duration from flower to hive must be included
                    self.next_reconnaissance_time = (int(
                        (self.next_reconnaissance_time + self.model.today(self.time_source_found)) / 2)
                                                     if abs(self.next_reconnaissance_time - self.model.today(self.time_source_found)) > 120 + self.targeted_flower.flight_duration
                                                     else int(self.model.steps + 120 + self.targeted_flower.flight_duration))

        elif self.homing_motivation > MAX_SEARCH_TIME:
            self.state = BeeState.RETURNING
            self.homing_motivation = 0

    def update_state_flying_to_search_area(self):
        """
        Updates the bee state when currently in state FLYING_TO_SEARCH_AREA
        """
        if self.accurate_position == self._search_area_center:
            self.state = BeeState.SEARCHING_ADVERTISED_SOURCE

    def update_state_loading(self):
        """
        Updates the bee state when currently in state LOADING_NECTAR
        """
        if self._remaining_time_in_state > 0:
            self._remaining_time_in_state -= 1

        else:
            self.loaded = True
            self.state = BeeState.RETURNING

    def update_state_returning(self):
        """
        Updates the bee state when currently in state RETURNING
        """
        if self.accurate_position == self.model.dance_floor:
            if self.loaded:
                self.state = BeeState.UNLOADING_NECTAR
                C = self.targeted_flower.sucrose_concentration
                self._remaining_time_in_state = int(
                    random.uniform(39 * (C ** 2) + 114.1 * C - 64.25, 159 * (C ** 2) - 140 * C + 166))

            else:
                self.state = BeeState.CLUSTERING
                self.accurate_position = self.model.dance_floor

    def update_state_unloading(self):
        """
        Updates the bee state when currently in state UNLOADING_NECTAR
        """
        if self._remaining_time_in_state > 0:
            self._remaining_time_in_state -= 1

        else:
            self.model.total_energy += NECTAR_REWARD * self.targeted_flower.sucrose_concentration
            self.loaded = False
            self.state = BeeState.DANCING
            self._remaining_time_in_state = 0.1713 * self.targeted_flower.value

    def update_state_dancing(self):
        """
        Updates the bee state when currently in state DANCING
        """
        if self._remaining_time_in_state > 0:
            self._remaining_time_in_state -= 1

        else:
            self.state = BeeState.PREPARING_TO_FLY_OUT
            self._remaining_time_in_state = random.randint(16, 51)

    def update_state_preparing_to_fly_out(self):
        """
        Updates the bee state when currently in state PREPARING_TO_FLY_OUT
        """
        if self._remaining_time_in_state > 0:
            self._remaining_time_in_state -= 1

        else:
            self.state = BeeState.FLYING_STRAIGHT_TO_FLOWER

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
            self.update_state_resting()

        # bee is CLUSTERING  ==> can see bees on the dance floor and can be seen by others on the dance floor
        elif self.state == BeeState.CLUSTERING:
            self.update_state_clustering()

        elif self.state == BeeState.WATCHING_WAGGLE_DANCE:
            self.update_state_watching_waggle_dance()

        elif self.state == BeeState.FLYING_STRAIGHT_TO_FLOWER or self.state == BeeState.SEARCHING_ADVERTISED_SOURCE:
            self.update_state_flying_to_source_or_searching()

        elif self.state == BeeState.FLYING_TO_SEARCH_AREA:
            self.update_state_flying_to_search_area()

        elif self.state == BeeState.LOADING_NECTAR:
            self.update_state_loading()

        elif self.state == BeeState.RETURNING:
            self.update_state_returning()

        elif self.state == BeeState.UNLOADING_NECTAR:
            self.update_state_unloading()

        elif self.state == BeeState.DANCING:
            self.update_state_dancing()

        elif self.state == BeeState.PREPARING_TO_FLY_OUT:
            self.update_state_preparing_to_fly_out()

        else:
            self.state = self.state



    def step(self) -> None:
        """
        Step function of the bee
        """
        if not isinstance(self.model, BeeForagingModel):
            raise AttributeError("BeeForagingModel is not an instance of BeeForagingModel")

        # small probability the agent dies at each step
        if random.random() < self._mortality_probability:
            self.model.agents.remove(self)

        self.update_status()

        # states in which bee is just losing energy
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

            # get random sample from all bees currently on the dance_floor
            seen_bees = BeeForagingModel.split_agents_by_percentage(self.model.get_bees_on_dance_floor(), 4, self)[0]

            for bee in seen_bees:
                if not bee.state == BeeState.DANCING:
                    continue

                if self.targeted_flower is None or self.targeted_flower == bee.targeted_flower:
                    self.targeted_flower = bee.targeted_flower
                    self.currently_watched_bee = bee
                    break


        elif self.state == BeeState.FLYING_STRAIGHT_TO_FLOWER:
            self.model.total_energy -= FLYING_COST_UNLOADED

            if Calc.get_distance(self.accurate_position, self.targeted_flower.location) <= FLYING_SPEED:
                self.accurate_position = self.targeted_flower.location

            else:
                self.move_bee_towards_point(self.targeted_flower.location, FLYING_SPEED)


        elif self.state == BeeState.FLYING_TO_SEARCH_AREA:
            self.model.total_energy -= FLYING_COST_UNLOADED
            if Calc.get_distance(self.accurate_position, self._search_area_center) <= FLYING_SPEED:
                self.accurate_position = self._search_area_center

            else:
                self.move_bee_towards_point(self._search_area_center, FLYING_SPEED)


        elif self.state == BeeState.SEARCHING_ADVERTISED_SOURCE:
            self.search(self.targeted_flower, self._search_area_center)


        elif self.state == BeeState.RETURNING:
            if self.loaded:
                self.model.total_energy -= FLYING_COST_LOADED

            else:
                self.model.total_energy -= FLYING_COST_UNLOADED


            if Calc.get_distance(self.accurate_position, self.model.dance_floor) <= FLYING_SPEED:
                self.accurate_position = self.model.dance_floor

            else:
                self.move_bee_towards_point(self.model.dance_floor, FLYING_SPEED)

    def __str__(self) -> str:
        return (
            f"Bee id: {self.unique_id}\n"
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

class DataCollector:
    """
    Collects data from a BeeForagingModel instance in specified intervals
    """
    def __init__(self, model : BeeForagingModel, path_to_csv :str, collection_interval : int) -> None:
        if not isinstance(model, BeeForagingModel):
            raise TypeError("BeeForagingModel must be a BeeForagingModel")

        if not os.path.exists(path_to_csv):
            raise AttributeError(f"File {path_to_csv} does not exist")

        self.model = model
        self.path_to_csv = path_to_csv
        self.collection_interval = collection_interval
        self.columns = ['number_of_starting_foragers', 'source_distance', 'sucrose_concentration', 'anticipation_method', 'time_step', 'energy']


        file_is_empty = not os.path.exists(path_to_csv) or os.path.getsize(path_to_csv) == 0
        if file_is_empty:
            self.make_header()

    def make_header(self) -> None:
        """
        Creates a header for the csv file
        """
        with open (self.path_to_csv, "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self.columns)

    def collect_data(self) -> None:
        """
        Collects data of all columns specified in self.columns
        """
        row = [self.model.number_of_starting_bees,
               self.model.initial_source_distance,
               self.model.sucrose_concentration,
               self.model.anticipation_method,
               self.model.steps,
               self.model.total_energy]

        with open (self.path_to_csv, "a") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(row)

    def check_for_collection_call(self) -> None:
        """
        Checks if the model has reached a collection time
        """
        if self.model.steps == 0:
            return

        if self.model.steps % self.collection_interval == 0:
            self.collect_data()

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
    _LOGGER.info(f"Running model instance for parameters: {specific_params}")
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
    number_of_starting_foragers = [10]
    source_distance = [500]
    sucrose_concentration = [1]
    anticipation_method = [1]

    number_of_steps = 172800
    number_of_runs_per_combination = 20

    csv_path = r"/home/valentin-rexer/uni/UofM/abm_files/sigmas/run_out.csv"

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
            "anticipation_method": a,
            "collector_path" : csv_path,
            "collection_interval": STEPS_PER_DAY
        })

    print(parallel_run(
        mp.cpu_count() - 2,
        number_of_runs_per_combination,
        number_of_steps,
        params
    ))

if __name__ == "__main__":
    main(sys.argv)