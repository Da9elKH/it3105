from agents import Agent, ANNAgent, CNNAgent, MCTSAgent, RandomAgent
from networks import ANN, CNN
from mcts import MCTS
from misc import StateManager
from environments import HexGame, HexGUI
from tqdm import trange
from itertools import permutations
from typing import Dict
from config import App


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

        return self.stats

    def run_game(self, players: Dict[int, str]):
        self.environment.reset()
        while not self.environment.is_game_over:
            move, dist = self.agents[players[self.environment.current_player]].get_move(greedy=False)
            self.environment.play(move)
        return self.environment.result, self.environment.current_player


if __name__ == "__main__":

    environment = HexGame(size=4)
    topp = TOPP(environment=environment)
    #topp.add_agent("random", RandomAgent(environment=environment))
    topp.add_agent("ann0", ANNAgent(environment=environment, network=ANN.from_file("(1) ANN_S4_B0.h5")))
    #topp.add_agent("ann20", ANNAgent(environment=environment, network=ANN.from_file("(1) ANN_S4_B20.h5")))
    #topp.add_agent("ann40", ANNAgent(environment=environment, network=ANN.from_file("(1) ANN_S4_B40.h5")))
    #topp.add_agent("ann60", ANNAgent(environment=environment, network=ANN.from_file("(1) ANN_S4_B60.h5")))
    topp.add_agent("ann80", ANNAgent(environment=environment, network=ANN.from_file("(1) ANN_S4_B80.h5")))

    if App.config("topp.visualize") and False:
        gui = HexGUI(environment=environment)
        gui.run_visualization_loop(lambda: print(topp.tournament(25)))
    else:
        print(topp.tournament(500))