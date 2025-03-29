### definition of fixed global variables ###
STEPS_PER_HOUR = 3600
STEPS_PER_DAY = 24 * STEPS_PER_HOUR # amount of ticks in a full day
MAX_POST_COLLECTION_CLUSTERING_TIME = 4.5 * STEPS_PER_HOUR

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