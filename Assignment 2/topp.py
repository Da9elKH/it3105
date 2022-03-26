from agents import Agent, ANNAgent, CNNAgent
from environments.hex import HexGame
from tqdm import tqdm
from itertools import permutations
from networks import ANN, CNN

class TOPP:
    def __init__(self, environment: HexGame):
        self.environment = environment
        self.agents: dict[str, Agent] = {}

    def add_agent(self, name: str, agent: Agent):
        self.agents[name] = agent

    def tournament(self, rounds=50):
        stats = {}
        battles = list(permutations(self.agents.keys(), 2))

        for agent in self.agents.keys():
            stats[agent] = {"W": 0, "L": 0}

        for i in tqdm(range(len(battles))):
            player1, player2 = battles[i]

            for _ in range(rounds):
                result, winner = self.run_game(self.agents[player1], self.agents[player2])

                if winner == 1:
                    stats[player1]["W"] += 1
                    stats[player2]["L"] += 1
                else:
                    stats[player2]["W"] += 1
                    stats[player1]["L"] += 1

        return stats

    def run_game(self, player1: Agent, player2: Agent):
        self.environment.reset()
        player = {1: player1, -1: player2}

        while not self.environment.is_game_over:
            move, dist = player[self.environment.current_player].get_move(greedy=False)
            self.environment.play(move)

        return self.environment.result, self.environment.current_player


if __name__ == "__main__":
    env = HexGame(size=7)
    topp = TOPP(environment=env)

    #filenames = ["(1) _ANN_S4_B0.h5", "(1) _ANN_S4_B75.h5", "(1) _ANN_S4_B100.h5", "(1) _ANN_S4_B125.h5"]
    filenames = ["(1) ANN_S7_B175.h5", "(1) ANN_S7_B100.h5", "(1) ANN_S7_B50.h5", "(1) ANN_S7_B0.h5"]

    for filename in filenames:
        topp.add_agent(filename, ANNAgent(environment=env, network=ANN.from_file(filename)))

    print(topp.tournament(100))