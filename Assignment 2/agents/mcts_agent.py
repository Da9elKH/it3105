from config import App
from environment import Environment
from mcts.mcts import MCTS
from agent import Agent


class MCTSAgent(Agent):
    def __init__(self, environment: Environment = None, mcts: MCTS = None):
        self.mcts = mcts if mcts else MCTS.from_config(environment=environment)
        self.T = App.config("mcts.temperature")
        super().__init__(environment)

    def get_move(self, greedy=False):
        self.mcts.search()
        return super().get_move(greedy)

    @property
    def distribution(self):
        dist = self.mcts.distribution
        dist = dist ** (1 / self.T) / sum(dist ** (1 / self.T))
        return dist

    def register_environment_move_hook(self):
        self.environment.register_move_hook(self.mcts.move)

    def register_environment_reset_hook(self):
        self.environment.register_reset_hook(self.mcts.reset)