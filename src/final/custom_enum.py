from enum import Enum

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
