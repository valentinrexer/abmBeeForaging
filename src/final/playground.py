import random
from enum import Enum

class Action(Enum):
    ACTION_1 =1,
    ACTION_2 =2,

x = Action.ACTION_2

match x:
    case Action.ACTION_1:
        print(1)

    case Action.ACTION_2:
        print(2)