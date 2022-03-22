from misc.state_manager import StateManager
from actor import Actor
from .node import Node
from misc.types import Move
from misc.state_manager import StateManager
from agents.agent import Agent
import random
import numpy as np
import time
import graphviz
from typing import Callable


class MCTS:
    def __init__(self, environment: StateManager, rollout_policy_agent: Agent, time_budget=2.0, c=1.0, epsilon=1.0):
        self.environment = environment.copy()
        self._reset_environment = environment.copy()
        self.root = Node(player=self.environment.current_player)

        self._agent = rollout_policy_agent
        self.time_budget = time_budget
        self.c = c
        self.epsilon = epsilon

    """ POLICIES """
    @staticmethod
    def _tree_policy(nodes: list[Node], c=1.0):
        move_values = [node.value(c=c) for node in nodes]
        best_node_index = np.argmax(move_values)
        node = nodes[best_node_index]
        return node

    def _rollout_policy(self, environment: StateManager, epsilon=1.0):
        if random.random() <= epsilon:
            return random.choice(environment.legal_moves)
        else:
            self._agent.environment = environment
            move, _ = self._agent.get_move(greedy=True)
            return move

    """ PROCESSES """

    def search(self):
        start_time = time.time()
        num_rollouts = 0

        while time.time() - start_time < self.time_budget:
            # Select a node to expend
            node, environment = self.selection(c=self.c)

            # Do expansion if possible
            node, environment = self.expand(node, environment)

            # Rollout a game
            winner = self.rollout(environment)

            # Backup the results
            self.backup(node, winner)
            num_rollouts += 1

    def selection(self, c=1.0):
        """
        Select a single node in the three to perform a simulation from
        """
        node = self.root
        environment = self.environment.copy()
        children = node.children

        while len(children) != 0:
            node = self._tree_policy(children, c=c)
            children = node.children
            environment.execute(node.move)

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
            node.children.append(Node(node, move, player=environment.next_player))

        random_child = random.choice(node.children)
        environment.execute(random_child.move)

        return random_child, environment

    def rollout(self, environment: StateManager):
        """
        Simulate a game based on the rollout policy and return the winning player
        """
        environment = environment.copy()

        while not environment.is_game_over:
            action = self._rollout_policy(environment, epsilon=self.epsilon)
            environment.execute(action)

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
        print(dist)
        dist /= sum(dist)

        return dist

    def move(self, move: Move):
        for child in self.root.children:
            if child.move == move:
                self.root = child
                self.root.parent = None
                self.environment.execute(move)

                print(f"Moved to {move}")
                return

        print(f"Failed to move to {move}")
        self.root = Node(parent=None, move=move, player=self.environment.next_player)
        self.environment.execute(move)
        return

    def reset(self, environment: StateManager = None):
        self.environment = environment.copy() if environment else self._reset_environment.copy()
        self.root = Node(player=self.environment.current_player)
        # TODO: Fix agent here?

    def draw(self):
        g = graphviz.Digraph('G', filename='MCTS.gv')
        node_count = 0
        queue = [self.root]
        while queue:
            parent = queue.pop(0)
            if parent.children:
                for child in parent.children:
                    child_info = "%d, %d, %d" % (child.N, child.Q, child.player)
                    parent_info = "%d, %d, %d" % (parent.N, parent.Q, parent.player)
                    g.edge(parent_info, child_info, label=str(child.move), id=str(node_count))
                    node_count += 1
                    queue.append(child)
        g.view()
