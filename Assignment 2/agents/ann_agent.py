from agents import Agent
from networks import ANN
from misc import StateManager, LiteModel
from config import App
import numpy as np


class ANNAgent(Agent):
    def __init__(self, environment: StateManager = None, network: ANN = None):
        super(ANNAgent, self).__init__(environment)
        self.network = network
        self.T = App.config("ann.temperature")

    @property
    def distribution(self):
        if isinstance(self.network.model, LiteModel):
            dist = self.network.predict(self.environment.ann_state)
        else:
            dist = self.network.predict(np.array([self.environment.ann_state]))[0].numpy()

        dist = dist * self.environment.legal_binary_moves
        dist = dist**(1/self.T) / sum(dist**(1/self.T))
        return dist

    def state_fc(self, environment, rotate=False):
        if rotate:
            return  environment.rotated_ann_state
        else:
            return environment.ann_state