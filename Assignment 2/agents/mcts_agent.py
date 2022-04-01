from .agent import Agent
from .random_agent import RandomAgent
from misc.state_manager import StateManager
from mcts.mcts import MCTS


class MCTSAgent(Agent):
    def __init__(self, environment: StateManager = None, mcts: MCTS = None, **mcts_params):
        self.mcts = mcts if mcts else MCTS(
            environment=environment,
            **{
                "use_time_budget": False,
                "rollouts": 1000,
                "time_budget": 1,
                "epsilon": 1.00,
                "c": 1.4,
                "rollout_policy_agent": RandomAgent(),
                **mcts_params
            }
        )
        super().__init__(environment)

    def get_move(self, greedy=False):
        self.mcts.search()
        return super().get_move(greedy)

    @property
    def distribution(self):
        return self.mcts.distribution

    def register_environment_move_hook(self):
        self.environment.register_move_hook(self.mcts.move)

    def register_environment_reset_hook(self):
        self.environment.register_reset_hook(self.mcts.reset)