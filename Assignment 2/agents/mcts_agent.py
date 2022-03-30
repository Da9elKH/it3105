from .agent import Agent
from .random_agent import RandomAgent
from misc.state_manager import StateManager
from mcts.mcts import MCTS


class MCTSAgent(Agent):
    def __init__(self, model: MCTS = None, environment: StateManager = None, mcts_params={}):
        super().__init__(environment)
        self.model = model if model else MCTS(
            environment=environment,
            **{
                "rollouts": 1000,
                "time_budget": 1,
                "epsilon": 1.00,
                "verbose": False,
                "c": 1.0,
                "rollout_policy_agent": RandomAgent(),
                **mcts_params
            }
        )

    def get_move(self, greedy=False):
        self.model.search()
        return super().get_move(greedy)

    def move(self, move):
        self.model.move(move)

    def reset(self):
        self.model.reset()

    @property
    def distribution(self):
        return self.model.distribution
