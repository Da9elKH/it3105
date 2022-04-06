from itertools import permutations
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from tabulate import tabulate
from tqdm import trange

from agent import Agent
from environment import Environment


class TOPP:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.agents: Dict[str, Agent] = {}
        self.stats: Dict[str, Dict[str, int]] = {}
        self.stats_idx = {}

    def add_agent(self, name: str, agent: Agent):
        self.agents[name] = agent
        self.stats[name] = {"W": 0, "L": 0}
        self.stats_idx[name] = len(self.agents) - 1

    def tournament(self, rounds=50):
        stats = np.zeros(shape=(len(self.agents),) * 2, dtype=np.int32)
        battles = list(permutations(self.agents.keys(), 2))
        id_transform = lambda x: tuple([self.stats_idx[name] for name in x])

        with trange(len(battles)) as t:
            for i in t:
                player1, player2 = battles[i]
                players = {1: player1, -1: player2}

                t.set_description(f"{player1} vs. {player2}")

                for _ in range(rounds):
                    result, winner = self.run_game(players)
                    stats[id_transform(battles[i])[::result]] += 1

                    # Save stats for this game
                    self.stats[players[winner]]["W"] += 1
                    self.stats[players[winner*(-1)]]["L"] += 1

        self.print_stats(stats, rounds)

    def run_game(self, players: Dict[int, str]):
        self.environment.reset()
        while not self.environment.is_game_over:
            move, dist = self.agents[players[self.environment.current_player]].get_move(greedy=False)
            self.environment.play(move)
        return self.environment.result, self.environment.current_player

    def print_stats(self, stats, rounds):
        info = {"Agents": list(self.agents.keys()), **{name: [] for name in self.agents.keys()}, "LOSS": []}
        for key_1, idx_1 in self.stats_idx.items():
            for key_2, idx_2 in self.stats_idx.items():
                info[key_1].append(stats[(idx_1, idx_2)])

        info["Agents"].append("WINS")
        for key, value in self.stats.items():
            info[key].append(f"({value['W']})")
            info["LOSS"].append(f"({value['L']})")

        print("\n---- TOPP STATISTICS ----\n")
        print(tabulate(info, headers='keys'))

        sbn.heatmap(stats/(2*rounds), xticklabels=list(self.agents.keys()), yticklabels=list(self.agents.keys()), annot=True)
        plt.show()