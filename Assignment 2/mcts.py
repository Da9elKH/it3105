from misc.state_manager import StateManager
from misc.types import Move
from collections import defaultdict
from typing import TypeVar, Generic, Optional
import random
import numpy as np
import graphviz
from copy import copy
import time
from actor import Actor

TNode = TypeVar("TNode", bound="Node")


class Node(Generic[TNode]):
    def __init__(self, parent: Optional[TNode] = None, move: Optional[Move] = None, player: int = 1):
        self.parent = parent
        self.move = move
        self.children = []
        self.player = player

        self.N = 0
        self.Q = 0

    def value(self, c=1.0):
        if self.N == 0:
            return np.inf
        return (self.Q / self.N) + c * np.sqrt(np.log(self.parent.N) / (self.N + 1))

    def visit(self):
        self.N += 1

    def update_value(self, winner: int):
        self.Q += -1 if winner == self.player else 1

    def distribution(self):
        return dict([(child.move, child.N / self.N) for child in self.children])


class MCTSAgent:
    def __init__(self, environment: StateManager, actor: Actor):
        self.environment = environment.copy()
        self._reset_environment = environment.copy()
        self.actor = actor
        self.root = Node(player=self.environment.current_player)

    """ POLICIES """
    @staticmethod
    def _tree_policy(nodes: list[Node], environment: StateManager, c=1.0):
        move_values = [node.value(c=c) for node in nodes]
        best_node_index = np.argmax(move_values) # if environment.current_player == 1 else np.argmin(move_values)
        node = nodes[best_node_index]
        return node

    def _rollout_policy(self, environment: StateManager, epsilon=1.0):
        if random.random() <= epsilon:
            return random.choice(environment.legal_moves)
        else:
            return self.actor.best_move(environment)

    """ PROCESSES """

    def search(self, time_budget=2.0, epsilon=1.0, c=1.0):
        start_time = time.time()
        num_rollouts = 0

        while time.time() - start_time < time_budget:
            # Select a node to expend
            node, environment = self.selection(c=c)

            # Do expansion if possible
            node, environment = self.expand(node, environment)

            # Rollout a game
            winner = self.rollout(environment, epsilon=epsilon)

            # Backup the results
            self.backup(node, winner)
            num_rollouts += 1

        return self.distribution, self.best_move

    def selection(self, c=1.0):
        """
        Select a single node in the three to perform a simulation from
        """
        node = self.root
        environment = self.environment.copy()
        children = node.children

        while len(children) != 0:
            node = self._tree_policy(children, environment, c=c)
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

    def rollout(self, environment: StateManager, epsilon=1.0):
        """
        Simulate a game based on the rollout policy and return the winning player
        """
        environment = environment.copy()

        while not environment.is_game_over:
            action = self._rollout_policy(environment, epsilon=epsilon)
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
        dist = dist/sum(dist)
        return dist

    @property
    def best_move(self):
        return self.environment.transform_binary_move_index_to_move(
            random.choice(np.argwhere(self.distribution == np.max(self.distribution)).flatten()))

    def move(self, move: Move):
        for child in self.root.children:
            if child.move == move:
                self.root = child
                self.environment.execute(move)
                self.root.parent = None

    def reset(self):
        self.environment = self._reset_environment.copy()
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
