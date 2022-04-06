from .agent import Agent
from environments import Environment
from misc import LiteModel
from networks.cnn import CNN
from config import App
import numpy as np


class CNNAgent(Agent):
    def __init__(self, environment: Environment = None, network: CNN = None):
        super(CNNAgent, self).__init__(environment)
        self.network = network
        self.T = App.config("cnn.temperature")

    @property
    def distribution(self):
        if isinstance(self.network.model, LiteModel):
            policy = self.network.predict(self.environment.cnn_state)
        else:
            policy = self.network.predict(np.array([self.environment.cnn_state])).numpy()

        if self.environment.current_player == -1:
            dist = policy.reshape(self.environment.state.shape).T.flatten()
        else:
            dist = policy.flatten()

        dist = dist * self.environment.legal_binary_moves
        dist = dist**(1/self.T) / sum(dist**(1/self.T))
        return dist

    @staticmethod
    def state_fc(environment: Environment, rotate=False):
        if rotate:
            return environment.rotated_cnn_state
        else:
            return environment.cnn_state
