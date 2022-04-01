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
            move, dist = self.agents[players[self.environment.current_player]].get_move(greedy=True)
            self.environment.play(move)
        return self.environment.result, self.environment.current_player


if __name__ == "__main__":
    env = HexGame(size=App.config("hex.size"))
    topp = TOPP(environment=env)
    gui = HexGUI(environment=env)

    mcts = MCTS(environment=env, use_time_budget=False, rollouts=1000, rollout_policy_agent=None)
    agent = MCTSAgent(environment=env, mcts=mcts)

    topp.add_agent("mcts", agent)
    topp.add_agent("random", RandomAgent(environment=env))

    if App.config("topp.matches"):
        gui.run_visualization_loop(lambda: print(topp.tournament(3)))
    else:
        print(topp.tournament(10))