from .agent import Agent
from misc.state_manager import StateManager
from networks.cnn import CNN
import numpy as np


class CNNAgent(Agent):
    def __init__(self, environment: StateManager = None, network: CNN = None):
        super(CNNAgent, self).__init__(environment)
        self.network = network
        self._value = None

    @property
    def distribution(self):
        policy, value = self.network.predict(np.array([self.environment.cnn_state]))
        self._value = value

        if self.environment.current_player == -1:
            dist = policy.numpy().reshape(self.environment.state.shape).T.flatten()
        else:
            dist = policy.numpy().flatten()

        dist = dist * self.environment.legal_binary_moves

        # dist = dist**(1/T) ()

        dist = dist / sum(dist)
        return dist

    @property
    def value(self):
        raise ValueError("NOT IMPLEMENTED")
        if self._value:
            return self._value
        else:
            return self.network.predict(np.array([self.environment.cnn_state]))[1]
