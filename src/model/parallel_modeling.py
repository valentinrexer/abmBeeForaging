#packages for modeling
import math
import sys

import mesa
from mesa.time import SimultaneousActivation

#packages for feature implementation
from enum import Enum
import random

#packages for multicore processing
import multiprocessing as mp

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
SEARCHING_ANGLE_RANGE = (-120.0, 120.0) #°

MAX_SEARCH_TIME = 960 #s [or ticks]
MAX_DANCE_ERROR = 30 #m

NECTAR_REWARD = 353.8 #C


### definition of model specific parameters ###
MAX_SIGHT = 1
MIN_BORDER_DISTANCE = 3
MAX_SEARCH_RADIUS = 50  #m max radius in which bee looks for the source


### definition of normal distribution variables for random drawing ###
MEAN = 0
STANDARD_DEVIATION = 20


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

        self.status = BeeStatus.RESTING
        self.targeted_position = self.model.grid.hive
        self.last_angle = 0.0
        self.search_area_center = self.model.grid.hive



    #todo: include energy loss function


    def search(self, destination, search_area):
        """
        Models the bee search behaviour
            => If the bee is closer than MAX_SIGHT  to the source it or was in this range sometimes between the current and last time step it directly flies to the flower
            => If the bee

        :param destination: the position the bee tries to find
        :param search_area: a tuple that describes the area in which the bee tries to find the destination
        """
        self.search_area_center = search_area

        if get_distance(self.accurate_position, destination) <= MAX_SIGHT or self.last_move_crossed_flower_radius(self.last_angle, SEARCHING_SPEED):
            self.move_bee_towards_point(destination, SEARCHING_SPEED)

        elif get_distance(self.accurate_position, self.search_area_center) >= MAX_SEARCH_RADIUS:
            angle = get_angle(self.accurate_position, self.search_area_center)
            angle = random_deviate_angle_equally(angle, SEARCHING_ANGLE_RANGE[0], SEARCHING_ANGLE_RANGE[1])
            self.move_bee_with_angle(angle, SEARCHING_SPEED)

        else:
            custom_sd = (MAX_SEARCH_RADIUS - get_distance(self.accurate_position, self.search_area_center)) / MAX_SEARCH_RADIUS * SEARCHING_ANGLE_RANGE[1]

            angle = random_deviate_angle(self.last_angle, MEAN, self.model.STANDARD_DEVIATION, SEARCHING_ANGLE_RANGE[0], SEARCHING_ANGLE_RANGE[1])
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


    def update_status(self):
        if self.status is BeeStatus.FLYING_OUT and self.accurate_position is self.targeted_position:
            self.status = BeeStatus.SEARCHING_ADVERTISED_SOURCE

        elif self.status is BeeStatus.SEARCHING_ADVERTISED_SOURCE and self.accurate_position is self.model.flower_location:
            self.status = BeeStatus.LOADING_NECTAR

        else:
            self.status = self.status



    def step(self):
        self.update_status()

        if self.status is BeeStatus.FLYING_OUT:
            self.move_bee_towards_point(self.targeted_position, FLYING_SPEED)

        elif self.status is BeeStatus.SEARCHING_ADVERTISED_SOURCE:
            self.search(self.model.flower_location, self.targeted_position)

        else:
            return

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
    def __init__(self, source_distance,sd):
        super().__init__()

        # create schedule
        self.schedule = SimultaneousActivation(self)

        # create grid ==> grid will have a size of source_distance + 70
        self.grid = BeeGrid((source_distance + 70) * 2)

        #create flower (food source) and place it on the grid
        #todo: ask for actual sucrose concentration
        flower = FlowerAgent(1, self, 1000)
        self.schedule.add(flower)
        self.flower_location = generate_random_point(self.grid.hive[0], self.grid.hive[1], source_distance, 0.01)
        self.STANDARD_DEVIATION = sd





### model functions ###


def generate_random_point(origin_x, origin_y, target_distance, tolerance=0.1, max_attempts=10000):
    """

    :param origin_x:
    :param origin_y:
    :param target_distance:
    :param tolerance:
    :param max_attempts:
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

    return normalize_angle(angle)



def random_deviate_angle(current_angle, mean, standard_deviation, min_value, max_value, radians=False):
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



def run_bee_model_instance(params):

    distance, foragers, sd = params

    model_instance = BeeForagingModel(int(distance), sd)

    for k in range(int(foragers)):
        bee_agent = ForagerBeeAgent(k, model_instance, BeeForagerType.PERSISTENT, 1, model_instance.grid.hive)
        bee_agent.status = BeeStatus.FLYING_OUT
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
                if forager.status == BeeStatus.FLYING_OUT or forager.status == BeeStatus.SEARCHING_ADVERTISED_SOURCE:
                    forager.status = BeeStatus.FLYING_OUT
                    forager.accurate_position = model_instance.grid.hive

        for _ in range(960):
            model_instance.schedule.step()

        c = 0

        for forager in model_instance.schedule.agents:
            if isinstance(forager, ForagerBeeAgent):
                if forager.status is BeeStatus.LOADING_NECTAR:
                    c += 1

        found_at_day.append(c)

        if c == 500 or it >= 20:
            break

    return found_at_day

vals = [-1, -5,-10,-20,-25,-30,-40]

def parallel_run(num_processes, num_iterations, params):
    # Create a list of parameter tuples for each iteration
    param_list = [params for _ in range(num_iterations)]

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_bee_model_instance, param_list)

    return results


SD_S = [10, 80, 90, 100, 110]
all_overall = []
all_overall_cnt = []

for _ in range(7):
    all_overall.append(0.0)
    all_overall_cnt.append(0)



def __main__(args):
    for sd in SD_S:
        print(sd)

        for _ in range(2):

            num_processes = mp.cpu_count()-2
            num_iterations = 50

            # Define parameters as a tuple
            params = (900, 500, sd)  # (distance, foragers)

            results = parallel_run(num_processes, num_iterations, params)


            longest_try = 0

            for result in results:
                # print(result)
                if len(result) > longest_try:
                    longest_try = len(result)


            arranged_res = []

            for result in results:
                arranged_result = []

                for elem in result:
                    arranged_result.append(elem)

                for _ in range(longest_try - len(result)):
                    arranged_result.append(500)

                arranged_res.append(arranged_result)

            avg = []
            avg_count = []
            for _ in range(longest_try):
                avg.append(0.0)
                avg_count.append(0)

            for day in range(longest_try):
                for result in arranged_res:
                    if len(result) > day:
                        avg[day] += result[day]
                        avg_count[day] += 1


            for day in range(len(avg)):
                avg[day] /= avg_count[day]
                avg[day] = int(avg[day])

            print(avg)



            print()


        print()


    for l in range(len(all_overall)):
        all_overall[l] = all_overall[l] / all_overall_cnt[l]

    print("all overall")
    print(all_overall)


if __name__ == '__main__':
    __main__(sys.argv)

