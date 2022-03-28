from ActorClient import ActorClient
from agents.buffer_agent import BufferAgent
from agents.random_agent import RandomAgent
from mcts.mcts import MCTS
from agents.mcts_agent import MCTSAgent
from environments import HexGame, HexGUI
import numpy as np


class OnlineAgent(ActorClient):
    def __init__(self, auth, qualify):
        super().__init__(auth=auth, qualify=qualify)

        self.environment = HexGame(size=7)
        self.agent = MCTSAgent(
            environment=self.environment,
            mcts_params={
                "time_budget": 0.95,
                "epsilon": 1.0,
                "c": 1.0,
                "rollout_policy_agent": RandomAgent(),
                "verbose": True
            }
        )

    def run(self, mode='qualifiers', visualize=True):
        if visualize:
            gui = HexGUI(environment=self.environment)
            gui.run_visualization_loop(lambda: super(OnlineAgent, self).run(mode))
        else:
            super(OnlineAgent, self).run(mode)

    def handle_game_start(self, start_player):
        player = {1: 1, 2: -1}
        self.environment.reset(start_player=player[start_player])

    def handle_get_action(self, state):
        # CHECK IF OPPONENT HAS DONE A MOVE
        previous_state = self.environment.state
        state = np.array(state[1:]).reshape(self.environment.state.shape)
        state[state == 2] = -1
        opponent_moves = np.argwhere((state - previous_state) != 0).tolist()

        if len(opponent_moves) == 1:
            self.environment.play(opponent_moves[0])

        # DO A MOVE ON CURRENT STATE
        move, _ = self.agent.get_move(greedy=True)
        self.environment.play(move)

        return int(move[0]), int(move[1])

    def handle_game_over(self, winner, end_state):
        super().handle_game_over(winner, end_state)


if __name__ == "__main__":
    oa = OnlineAgent(auth="e1431af64ca24ffa9f2f3887e6b41a32", qualify=False)
    oa.run(visualize=True)
