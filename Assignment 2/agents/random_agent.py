from .agent import Agent
import numpy as np


class RandomAgent(Agent):

    @property
    def distribution(self):
        return np.array(self.environment.legal_binary_moves)/sum(self.environment.legal_binary_moves)
