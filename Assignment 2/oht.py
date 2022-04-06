from ActorClient import ActorClient
from agents import CNNAgent
from environments import Hex
from networks import CNN
from config import App
import numpy as np


class OHT(ActorClient):
    def __init__(self, auth, qualify, environment=None):
        super().__init__(auth=auth, qualify=qualify)
        self.environment = environment if environment else Hex(size=App.config("environment.size"))
        self.agent = CNNAgent(environment=self.environment, network=CNN.from_file(App.config("oht.agent")))

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
