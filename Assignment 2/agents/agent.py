from abc import ABC, abstractmethod
from misc.state_manager import StateManager
import numpy as np
import random


class Agent(ABC):
    def __init__(self, environment: StateManager):
        self.environment = environment

    def get_move(self, greedy=False):
        distribution = self.distribution

        if greedy:
            value = np.max(distribution)
        else:
            value = np.random.choice(distribution, p=distribution)

        return self.environment.transform_binary_move_index_to_move(
            random.choice(np.argwhere(distribution == value).flatten()))

    @property
    @abstractmethod
    def distribution(self):
        pass
