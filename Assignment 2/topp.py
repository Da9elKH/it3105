from agents import Agent, ANNAgent, CNNAgent, MCTSAgent
from misc import StateManager
from environments import HexGame, HexGUI
from tqdm import tqdm, trange
from itertools import permutations
from networks import ANN, CNN
from typing import Dict


class TOPP:
    def __init__(self, environment: StateManager):
        self.environment = environment
        self.agents: Dict[str, Agent] = {}
        self.stats: Dict[str, Dict[str, int]] = {}

    def add_agent(self, name: str, agent: Agent):
        self.agents[name] = agent
        self.stats[name] = {"W": 0, "L": 0}

    def tournament(self, rounds=50):
        battles = list(permutations(self.agents.keys(), 2))

        with trange(len(battles)) as t:
            for i in t:
                player1, player2 = battles[i]
                players = {1: player1, -1: player2}

                t.set_description(f"{player1} vs. {player2}")

                for _ in range(rounds):
                    result, winner = self.run_game(players)

                    # Save stats for this game
                    self.stats[players[winner]]["W"] += 1
                    self.stats[players[winner*(-1)]]["L"] += 1

    def run_game(self, players: Dict[int, str]):
        self.environment.reset()
        while not self.environment.is_game_over:
            move, dist = self.agents[players[self.environment.current_player]].get_move(greedy=False)
            self.environment.play(move)
        return self.environment.result, self.environment.current_player


if __name__ == "__main__":
    env = HexGame(size=7)
    topp = TOPP(environment=env)
    gui = HexGUI(environment=env)

    filenames = ["(1) CNN_S7_B5.h5", "(1) CNN_S7_B85.h5"]

    for filename in filenames:
        topp.add_agent(filename, CNNAgent(environment=env, network=CNN.from_file(filename)))

    gui.run_visualization_loop(lambda: topp.tournament(50))
