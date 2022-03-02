from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def action(self, state, valid_actions):
        pass
