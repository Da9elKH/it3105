from .node import Node
from misc import StateManager, Move
from agents import Agent
import random
import numpy as np
import time
import graphviz
from typing import List
from config import App
import logging
logger = logging.getLogger(__name__)
logger.setLevel(App.config("mcts.log_level"))


class MCTS:
    def __init__(self, environment: StateManager, rollout_policy_agent: Agent = None, use_time_budget=False, time_budget=0.0, rollouts=0, c=1.0,
                 epsilon=1.0):
        self.environment = environment
        self.root = Node(player=self.environment.current_player)

        self._agent = rollout_policy_agent

        self.use_time_budget = use_time_budget
        self.time_budget = time_budget
        self.rollouts = rollouts

        self.c = c
        self.epsilon = epsilon

        self.config = {
            "use_time_budget": use_time_budget,
            "rollouts": rollouts,
            "time_budget": time_budget,
            "c": c,
            "rp_agent": rollout_policy_agent.__class__.__name__,
            "epsilon": epsilon
        }

    """ POLICIES """

    @staticmethod
    def _tree_policy(nodes: List[Node], c=1.0):
        values = [node.value(c=c) for node in nodes]
        max_value = max(values)
        max_ids = [idx for idx, val in enumerate(values) if val == max_value]
        i = random.choice(max_ids)
        return nodes[i]

    def _rollout_policy(self, environment: StateManager):
        if random.random() <= self.epsilon or self._agent is None:
            return random.choice(environment.legal_moves)
        else:
            self._agent.environment = environment
            move, _ = self._agent.get_move(greedy=True)
            return move

    """ PROCESSES """

    def search(self):
        start_time = time.time()
        num_rollouts = 0

        while (time.time() - start_time < self.time_budget and self.use_time_budget) or (num_rollouts < self.rollouts and not self.use_time_budget):
            # Select a node to expend
            node, environment = self.selection(c=self.c)

            # Do expansion if possible
            node, environment = self.expand(node, environment)

            # Rollout a game
            winner = self.rollout(environment)

            # Backup the results
            self.backup(node, winner)
            num_rollouts += 1

        logger.info(f"Ran {num_rollouts} rollouts in {time.time() - start_time} seconds")

    def selection(self, c=1.0):
        """
        Select a single node in the three to perform a simulation from
        """
        node = self.root
        environment = self.environment.copy()
        children = node.children

        while len(children) != 0:
            node = self._tree_policy(list(children.values()), c=c)
            children = node.children
            environment.play(node.move)

        return node, environment

    @staticmethod
    def expand(node: Node, environment: StateManager):
        """
        Generate the children nodes of the passed parent node, and add them to the tree
        """
        # If the node is a terminal node
        if environment.is_game_over:
            return node, environment

        # If the node is not a terminal node
        for move in environment.legal_moves:
            node.children[move] = Node(node, move, player=environment.next_player)

        move, node = random.choice(list(node.children.items()))
        environment.play(move)

        return node, environment

    def rollout(self, environment: StateManager):
        """
        Simulate a game based on the rollout policy and return the winning player
        """

        while not environment.is_game_over:
            action = self._rollout_policy(environment)
            environment.play(action)

        return environment.current_player

    @staticmethod
    def backup(node: Node, winner: int):
        """
        Update the node statistics from the passed node to root, based on the rollout game
        """

        while node is not None:
            node.visit()
            node.update_value(winner)
            node = node.parent

    """ MISC """

    @property
    def distribution(self):
        node_action_distribution = self.root.distribution()
        dist = np.zeros_like(self.environment.legal_binary_moves, dtype=np.float32)

        for k, v in node_action_distribution.items():
            index = self.environment.transform_move_to_binary_move_index(k)
            dist[index] = v

        # Normalize the distribution
        dist /= sum(dist)

        return dist

    def move(self, move: Move):
        # Find or set new root node based on move
        if move in self.root.children:
            self.root = self.root.children[move]
        else:
            logger.debug(f"Child node {move} not when moving root")
            self.root = Node(move=move, player=self.environment.current_player)

        # Update the parent to be none.
        self.root.parent = None

    def reset(self):
        logger.debug("Reset MCTS to new node")
        self.root = Node(player=self.environment.current_player)

    def draw(self, only_visited=True):
        dot = graphviz.Digraph('G', filename='MCTS.gv')
        queue = [self.root]

        dot.node(str(id(self.root)), "N: %d\nQ: %d\n P:%d" % (self.root.N, self.root.Q, self.root.player))

        while queue:
            parent = queue.pop(0)
            if parent.children.values():
                for child in parent.children.values():
                    if child.N == 0 and only_visited:
                        continue

                    dot.node(str(id(child)), "N: %d\nQ: %d\n P:%d" % (child.N, child.Q, child.player))
                    dot.edge(str(id(parent)), str(id(child)), label=str(child.move))
                    queue.append(child)

        dot.view()
