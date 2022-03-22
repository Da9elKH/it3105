from ActorClient import ActorClient
from agents.agent import Agent
from agents.buffer_agent import BufferAgent
from agents.random_agent import RandomAgent
from mcts.mcts import MCTS
from agents.mcts_agent import MCTSAgent
from hex import GameWindow, HexGame
import numpy as np


class OnlineAgent(ActorClient):
    def __init__(self, auth, qualify):
        super().__init__(auth=auth, qualify=qualify)

        self.environment = HexGame(size=7)
        self.agent = MCTSAgent(
            environment=self.environment,
            model=MCTS(
                environment=self.environment,
                time_budget=1.0,
                epsilon=1.0,
                c=1.0,
                rollout_policy_agent=RandomAgent(environment=self.environment)
        ))

        self.buffer = None

    def handle_game_start(self, start_player):
        self.environment.start_player = start_player
        self.environment.reset()
        self.agent.model.reset(self.environment)
        self.buffer = BufferAgent()

    def handle_get_action(self, state):
        # Opponent move
        previous_state = self.environment.state
        state = np.array(state[1:]).reshape(self.environment.state.shape)


        print(state)

        print(np.argwhere((state - previous_state) != 0))

        opponent_move = tuple(np.argwhere((state - previous_state) != 0).tolist()[0])
        self.environment.execute(opponent_move)
        self.agent.model.move(opponent_move)

        # Current player move
        move, _ = self.agent.get_move(greedy=True)
        self.environment.execute(move)
        self.agent.model.move(move)

        self.buffer.add_move(opponent_move)
        self.buffer.add_move(move)

        return int(move[0]), int(move[1])

    #def handle_game_over(self, winner, end_state):
    #    #window = GameWindow(width=1000, height=600, game=HexGame(size=7), agent=self.buffer, view_update_rate=2.0)
    #    #window.run()
    #    super().handle_game_over(winner, end_state)


if __name__ == "__main__":
    oa = OnlineAgent(auth="e1431af64ca24ffa9f2f3887e6b41a32", qualify=False)
    oa.run()