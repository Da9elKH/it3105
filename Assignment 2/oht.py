from ActorClient import ActorClient
from agents import RandomAgent, MCTSAgent, CNNAgent, ANNAgent
from environments import HexGame, HexGUI
from networks import CNN, ANN
from config import App
from mcts import MCTS
import numpy as np


class OHT(ActorClient):
    def __init__(self, auth, qualify):
        super().__init__(auth=auth, qualify=qualify)
        self.environment = HexGame(size=7)
        self.agent = CNNAgent(environment=self.environment, network=CNN.from_file(App.config("oht.agent")))

    def run(self, mode='qualifiers'):
        if App.config("oht.visualize"):
            gui = HexGUI(environment=self.environment)
            gui.run_visualization_loop(lambda: super(OHT, self).run(mode))
        else:
            super(OHT, self).run(mode)

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
            self.environment.play(tuple(opponent_moves[0]))

        # DO A MOVE ON CURRENT STATE
        move, _ = self.agent.get_move(greedy=True)
        self.environment.play(move)

        return int(move[0]), int(move[1])

    def handle_game_over(self, winner, end_state):
        super().handle_game_over(winner, end_state)


if __name__ == "__main__":
    oa = OHT(auth=App.config("oht.auth"), qualify=App.config("oht.qualify"))
    oa.run(mode=App.config("oht.mode"))
