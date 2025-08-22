""" constant numeric values used in the simulation """

# time related constants
STEPS_PER_HOUR = 3600
STEPS_PER_DAY = 24 * STEPS_PER_HOUR
MAX_DURATION_UNTIL_CLUSTERING_TIME_AFTER_LAST_COLLECTION = 4.5 * STEPS_PER_HOUR

# constants related to flying behavior and flying energy costs
RESTING_ENERGY_COST = 6.2 * (1.0/1000) #mJ/s
UNLOADING_NECTAR_ENERGY_COST = 9.3 * (1.0/1000) #mJ/s
FLYING_COST_UNLOADED = 37.0 * (1.0/1000) #mJ/s
FLYING_COST_LOADED = 75 * (1.0/1000) #mJ/s
FLYING_SPEED = 6.944 #m/s

# constants related to searching behavior
SEARCHING_SPEED = FLYING_SPEED / 3 #m/s
SEARCHING_ANGLE_RANGE = (-120.0, 120.0) #°
MAX_SEARCH_TIME = 960 #s [or ticks]
MAX_SIGHT = 1 #m
MIN_BORDER_DISTANCE = FLYING_SPEED + 0.1 #m
MAX_SEARCH_RADIUS = 50 #m
NECTAR_REWARD = 353.8 #J

# constants affecting agent presence in the simulation
MORTALITY_PROBABILITY = 7.7 * 10 ** (-7)
DAY_SKIPPER_RATE = 0.20

# constants determining modeling of random behavior
DEFAULT_MEAN = 0
DEFAULT_STANDARD_DEVIATION = 35