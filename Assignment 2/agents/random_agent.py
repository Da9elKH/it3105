from agent import Agent
from random import choice


class RandomAgent(Agent):
    def action(self, state, valid_actions, game):
        return choice(valid_actions)
