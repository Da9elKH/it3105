from .agent import Agent
from misc.state_manager import StateManager
from mcts.mcts import MCTS


class MCTSAgent(Agent):
    def __init__(self, model: MCTS = None, environment: StateManager = None, mcts_params={}):
        super().__init__(environment)
        self.model = model if model else MCTS(environment=environment, **mcts_params)

    def get_move(self, greedy=False):
        self.model.search()
        return super().get_move(greedy)

    def play(self, move):
        self.model.move(move)

    @property
    def distribution(self):
        return self.model.distribution
