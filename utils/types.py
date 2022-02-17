from typing import Union
from enum import Enum

Action = Union[int, tuple[int, int]]
ActionList = list[Action]


class GameType(Enum):
    CART_POLE = 1
    TOWER_OF_HANOI = 2
    GAMBLER = 3
    AI_GYM = 4


class CriticType(Enum):
    TABLE = 1
    NEURAL_NETWORK = 2