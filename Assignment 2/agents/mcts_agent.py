from agent import Agent
from misc.state_manager import StateManager
from mcts.mcts import MCTS


class MCTSAgent(Agent):
    def __init__(self, environment: StateManager, model: MCTS):
        super().__init__(environment)
        self.model = model

    def get_move(self, greedy=False):
        self.model.search()
        return super().get_move(greedy)

    @property
    def distribution(self):
        return self.model.distribution
