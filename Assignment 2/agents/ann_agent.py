from .agent import Agent
import numpy as np
import random


class ANNAgent(Agent):
    def __init__(self, environment, network):
        super(ANNAgent, self).__init__(environment)
        self.network = network

    @property
    def distribution(self):
        dist = self.network.predict(np.array([self.environment.flat_state]))[0].numpy()
        dist = dist * self.environment.legal_binary_moves
        dist = dist / sum(dist)
        return dist
