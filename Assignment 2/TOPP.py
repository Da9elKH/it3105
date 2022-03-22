from agents.agent import Agent
from agents.ann_agent import ANNAgent
from agents.buffer_agent import BufferAgent
from ann import Network
from hex import HexGame, GameWindow
from tqdm import tqdm, trange
from itertools import permutations
import numpy as np

class TOPP:
    def __init__(self, environment: HexGame):
        self.environment = environment

    def tournament(self, agents: dict[str, Agent], rounds=50):
        stats = {}
        battles = list(permutations(agents.keys(), 2))

        for agent in agents.keys():
            stats[agent] = 0

        for i in tqdm(range(len(battles))):
            player1, player2 = battles[i]

            for _ in range(rounds):
                result, winner = self.run_game(agents[player1], agents[player2])
                stats[player1] += result
                stats[player2] += (-1)*result

        return stats

    def run_game(self, player1: Agent, player2: Agent):
        self.environment.reset()
        players = [player1, player2]

        #buffer = BufferAgent()
        #window = GameWindow(width=1000, height=600, game=HexGame(size=7), agent=buffer, view_update_rate=2.0)

        while not self.environment.is_game_over:
            index = self.environment.current_player - 1
            move, dist = players[index].get_move(greedy=False)
            self.environment.execute(move)

            meta_dist = {index: str(round(v * 100, 2)) + "%" for index, v in
                         np.ndenumerate(dist.reshape(env.state.shape))}
            #buffer.add_move(move)
            #buffer.add_distribution(meta_dist)

        #window.run()

        return self.environment.result, self.environment.current_player


if __name__ == "__main__":
    agents = {}
    env = HexGame(size=4)
    topp = TOPP(environment=env)

    files = ["(1) S4_B0.h5", "(1) S4_B225.h5"]
    for filename in files:
        agents[filename] = ANNAgent(environment=env, network=Network.from_file(filename))

    print(topp.tournament(agents, 50))