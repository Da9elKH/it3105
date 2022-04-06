from agents import Agent
from networks import ANN
from environments import Environment
from misc import LiteModel
from config import App
import numpy as np


class ANNAgent(Agent):
    def __init__(self, environment: Environment = None, network: ANN = None):
        super(ANNAgent, self).__init__(environment)
        self.network = network
        self.T = App.config("ann.temperature")

    @property
    def distribution(self):
        if isinstance(self.network.model, LiteModel):
            policy = self.network.predict(self.environment.ann_state)
        else:
            policy = self.network.predict(np.array([self.environment.ann_state]))[0].numpy()

        if self.environment.current_player == -1:
            dist = policy.reshape(self.environment.state.shape).T.flatten()
        else:
            dist = policy

        dist = dist * self.environment.legal_binary_moves
        dist = dist**(1/self.T) / sum(dist**(1/self.T))
        return dist

    @staticmethod
    def state_fc(environment: Environment, rotate=False):
        if rotate:
            return environment.rotated_ann_state
        else:
            return environment.ann_state
