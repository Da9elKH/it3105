from abc import ABC, abstractmethod
from misc import StateManager, Move
import numpy as np
import random


class Agent(ABC):
    def __init__(self, environment: StateManager = None):
        self.environment = environment
        if self.environment:
            self.environment.register_reset_hook(self.reset)
            self.environment.register_move_hook(self.move)

    def get_move(self, greedy=False):
        distribution = self.distribution

        if greedy:
            value = np.max(distribution)
        else:
            value = np.random.choice(distribution, p=distribution)

        return self.environment.transform_binary_move_index_to_move(
            random.choice(np.argwhere(distribution == value).flatten())), distribution

    @property
    @abstractmethod
    def distribution(self):
        pass

    def reset(self):
        pass

    def move(self, move: Move):
        pass
