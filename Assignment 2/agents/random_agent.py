from agent import Agent
from random import choice
import numpy as np


class RandomAgent(Agent):

    @property
    def distribution(self):
        return np.array(self.environment.legal_binary_moves)/sum(self.environment.legal_binary_moves)
