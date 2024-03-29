import numpy as np

from ActorClient import ActorClient
from agents import CNNAgent
from config import App
from environments import Hex
from networks import CNN


class OHT(ActorClient):
    def __init__(self, auth, qualify, environment=None, agent=None):
        super().__init__(auth=auth, qualify=qualify)
        self.environment = environment if environment else Hex(size=7)
        self.agent = agent if agent else CNNAgent(environment=self.environment, network=CNN.from_file("7x7/(1) CNN_S7_B1638.h5"))

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
