
from typing import TypeVar, Generic, Optional, Tuple, Dict, List, Callable
import random
import time
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log
from functools import lru_cache
from unionfind import UnionFind


cdef tuple PLAYERS = (1, -1)

THexGame = TypeVar("THexGame", bound="HexGame")

cdef class HexGame:
    cdef public:
        int size
        int current_player
        int _start_player
        dict _uf
        bint _is_game_over
        set _legal_moves
        np.ndarray _shadow_state
        list _reset_hooks
        list _move_hooks

    def __init__(self, size=5, start_player=PLAYERS[0], state=np.zeros((1,))):
        super().__init__()
        self.size = size
        self._start_player = start_player
        self._uf = self._uf_init()

        # These will be updated in state.setter
        self._is_game_over = False
        self._legal_moves = set([])

        self._shadow_state = self._state_init()
        self.state = state

        self._reset_hooks: List[Callable] = []
        self._move_hooks: List[Callable] = []

    # ####################
    #    State handling
    # ####################
    @property
    def state(self):
        return self._shadow_state[1:-1, 1:-1]

    @state.setter
    def state(self, value):
        self._shadow_state[1:-1, 1:-1] = value
        self._on_state_updated()

    @property
    def flat_state(self):
        return [self.current_player, *self.state.flatten()]

    @property
    def rotated_cnn_state(self):
        return self._cnn_state(True)

    @property
    def cnn_state(self):
        return self._cnn_state(False)

    cdef np.ndarray _cnn_state(self, bint rotate=False):
        cdef tuple size = self.state.shape
        cdef np.ndarray cnn_state = self.state if not rotate else self.state[::-1, ::-1]
        cdef np.ndarray player1 = (cnn_state == PLAYERS[0])
        cdef np.ndarray player2 = (cnn_state == PLAYERS[1])
        cdef np.ndarray empty = (cnn_state == 0)
        cdef list to_play = []

        if self.current_player == PLAYERS[0]:
            to_play = [np.ones(size), np.zeros(size)]
        else:
            to_play = [np.zeros(size), np.ones(size)]

        return np.moveaxis(
            np.array([player1, player2, empty, *to_play]), 0, 2)

    cdef void _on_state_updated(self):
        self._uf_state_sync()

        # Check if the last move ended the game
        cdef int last_player = self._current_player * (-1)
        self._is_game_over = self._uf[last_player].connected("start", "end")

        # Set possible moves
        self._legal_moves = set([tuple(x) for x in np.argwhere(self.state == 0).tolist()])

        # If not then update the player
        if self.is_game_over:
            self.current_player = last_player
        else:
            self.current_player = last_player * (-1)

    cdef np.ndarray _state_init(self):
        cdef np.ndarray shadow_state = np.zeros((self.size + 2, self.size + 2), dtype=np.double)
        shadow_state[:, 0] = shadow_state[:, -1] = PLAYERS[1]
        shadow_state[0, :] = shadow_state[-1, :] = PLAYERS[0]
        return shadow_state

    # ####################
    #    Game Playing
    # ####################
    cpdef void play(self, tuple move):
        if isinstance(move, list):
            move = tuple(move)

        if move in self.legal_moves and not self.is_game_over:
            # Register the move
            self.state[move] = self.current_player
            self._legal_moves.remove(move)

            # Sync uf neighbors
            self._uf_merge_neighbors(move, self.current_player)

            # Update game status
            self._is_game_over = self.uf.connected("start", "end")

            if not self.is_game_over:
                # Switch player
                self.current_player = self.next_player

            # Broadcast move to agents following this state
            self.broadcast_move(move)
        else:
            raise ValueError(f"Move {move} is not allowed")

    @property
    def legal_moves(self):
        if self.is_game_over:
            return []
        return list(self._legal_moves)

    @property
    def legal_binary_moves(self):
        return np.logical_not(self.state).flatten().tolist()

    cpdef int transform_move_to_binary_move_index(self, tuple move):
        return move[0] * self.size + move[1]

    cpdef tuple transform_binary_move_index_to_move(self, int binary_move_index):
        return np.unravel_index(binary_move_index, shape=self.state.shape)

    @property
    def next_player(self):
        return self.current_player * (-1)

    @property
    def _current_player(self):
        return self._start_player * (-1) ** (np.sum(self.state != 0))

    # ####################
    #   Game over check
    # ####################
    @property
    def is_game_over(self):
        return self._is_game_over

    @property
    def result(self):
        return self.current_player

    @property
    def uf(self) -> UnionFind:
        return self._uf[self.current_player]

    cdef void _uf_state_sync(self):
        # Sync with current state
        cdef int player = 1
        cdef list location = None

        for player in PLAYERS:
            for location in np.argwhere(self.state == player).tolist():
                self._uf_merge_neighbors(tuple(location), player)

    cdef list NEIGHBORS(self, int r, int c):
        return [(r + 1, c - 1), (r, c - 1), (r - 1, c), (r - 1, c + 1), (r, c + 1), (r + 1, c)]

    cdef list _uf_neighbors(self, tuple location, int size):
        return [(i, j) for i, j in self.NEIGHBORS(location[0], location[1]) if 0 <= i <= size + 1 and 0 <= j <= size + 1]

    cdef void _uf_merge_neighbors(self, tuple location, int player):
        # Merge all neighbors
        location = (location[0] + 1, location[1] + 1)
        cdef list neighbors = self._uf_neighbors(location, self.size)
        cdef tuple neighbor = None

        for neighbor in neighbors:
            if self._shadow_state[location] == self._shadow_state[neighbor]:
                self._uf[player].union(location, neighbor)

    cdef object _uf_init(self):
        # Initiate union-find for both players
        cdef int i = 0
        cdef dict uf = {
            PLAYERS[0]: UnionFind(["start", "end"]),
            PLAYERS[1]: UnionFind(["start", "end"])
        }

        # Connect player edges
        for i in range(self.size + 2):
            uf[PLAYERS[0]].union("start", (0, i))
            uf[PLAYERS[0]].union("end", (self.size + 1, i))
            uf[PLAYERS[1]].union("start", (i, self.size + 1))
            uf[PLAYERS[1]].union("end", (i, 0))

        return uf

    # ####################
    #        Misc
    # ####################
    def copy(self):
        new = self.__class__(start_player=self._start_player, size=self.size, state=self.state.copy())
        return new

    def reset(self, start_player=None):
        self._start_player = start_player if start_player else self._start_player
        self._uf = self._uf_init()
        self.state = 0
        self.broadcast_reset()


    def broadcast_reset(self):
        [hook() for hook in self._reset_hooks]

    def broadcast_move(self, move: tuple):
        [hook(move) for hook in self._move_hooks]

    def register_reset_hook(self, function: Callable):
        self._reset_hooks.append(function)

    def register_move_hook(self, function: Callable):
        self._move_hooks.append(function)



cdef class Node:
    cdef public:
        Node parent
        tuple move
        dict children
        int player
        int N
        int Q

    def __init__(self, Node parent = None, tuple move = None, int player = 1):
        self.parent = parent
        self.move = move
        self.children = {}
        self.player = player

        self.N = 0
        self.Q = 0

    cpdef double value(self, double c = 1.0):
        cdef double score = 0.0 if self.N == 0 else (self.Q / self.N)
        cdef double explr = c * sqrt(log(self.parent.N) / (self.N + 1))
        return score + explr

    cdef void visit(self):
        self.N += 1

    cdef void update_value(self, int winner):
        self.Q += -1 if winner == self.player else 1

    cdef dict distribution(self):
        return {move: child.N/self.N for move, child in self.children.items()}


cdef class MCTS:

    cdef public:
        HexGame environment
        HexGame _reset_environment
        Node root
        double time_budget
        double c
        int rollouts
        bint verbose
        bint use_time_budget      

    def __init__(self, HexGame environment, bint use_time_budget=False, double time_budget = 0.0, int rollouts = 0, double c = 1.0, bint verbose = False):
        self.environment = environment
        self.root = Node(player=self.environment.current_player)
        self.time_budget = time_budget
        self.rollouts = rollouts
        self.c = c
        self.verbose = verbose

    """ POLICIES """
    cdef Node _tree_policy(self, list nodes, double c = 1.0):
        cdef Node node = None
        cdef list values = []
        cdef int i = 0
        cdef int leng = len(nodes)

        for i in range(leng):
            values.append(nodes[i].value(c=c))        
        
        cdef double max_value = max(values)
        cdef list max_ids = [idx for idx, val in enumerate(values) if val == max_value]
        i = random.choice(max_ids)
        return nodes[i]

    cpdef tuple _rollout_policy(self, HexGame environment):
        return random.choice(environment.legal_moves)

    """ PROCESSES """

    cpdef void search(self):
        cdef int start_time = time.time()
        cdef int num_rollouts = 0
        cdef Node node = None
        cdef HexGame environment = None
        cdef int winner = 0

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

        
        print(f"MCTS: Ran {num_rollouts} rollouts in {time.time() - start_time} seconds")

    cdef tuple selection(self, double c=1.0):
        """
        Select a single node in the three to perform a simulation from
        """
        cdef Node node = self.root
        cdef HexGame environment = self.environment.copy()
        cdef dict children = node.children

        while len(children) != 0:
            node = self._tree_policy(list(children.values()), c=c)
            children = node.children
            environment.play(node.move)

        return node, environment

    cdef tuple expand(self, Node node, HexGame environment):
        """
        Generate the children nodes of the passed parent node, and add them to the tree
        """
        # If the node is a terminal node
        if environment.is_game_over:
            return node, environment

        cdef tuple move = None
        cdef int i = 0
        cdef int leng = len(environment.legal_moves)

        # If the node is not a terminal node
        for i in range(leng):
            move = environment.legal_moves[i]
            node.children[move] = Node(node, move, player=environment.next_player)
        
        move, node = random.choice(list(node.children.items()))
        environment.play(move)

        return node, environment

    cdef int rollout(self, HexGame environment):
        """
        Simulate a game based on the rollout policy and return the winning player
        """
        cdef tuple action = None

        # environment = environment.copy()
        while not environment.is_game_over:
            action = self._rollout_policy(environment)
            environment.play(action)

        return environment.current_player

    cdef void backup(self, Node node, int winner):
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

    def move(self, move: tuple):
        # Find or set new root node based on move
        if move in self.root.children:
            self.root = self.root.children[move]
        else:
            self.root = Node(move=move, player=self.environment.current_player)

        # Update the parent to be none.
        self.root.parent = None

    def reset(self):
        self.root = Node(player=self.environment.current_player)
