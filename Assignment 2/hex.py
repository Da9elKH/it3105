from collections import deque
from functools import lru_cache
from typing import Union

import arcade
import arcade.gui
import numpy as np
import networkx as nx
import math
from agents.agent import Agent
from unionfind import UnionFind
from misc.state_manager import StateManager
from copy import deepcopy
from typing import TypeVar, Generic
import config
from itertools import product

RED_COLOR = "fc766a"
RED_COLOR_LIGHT = "fed1cd"
BLUE_COLOR = "5b84b1"
BLUE_COLOR_LIGHT = "bacbde"
PLAYERS = (1, -1)

THexGame = TypeVar("THexGame", bound="HexGame")


class NewHexGame:
    def __init__(self, size=5, start_player=PLAYERS[0], state=np.zeros((1,), dtype=np.int8)):
        self.size = size
        self._start_player = start_player
        self._uf = self._uf_init()

        # These will be updated in state.setter
        self.current_player = None
        self._is_game_over = None

        self._shadow_state = self._state_init()
        self.state = state

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
    def cnn_state(self):
        player_state = (self.state == self.current_player).astype(np.int32)
        opponent_state = (self.state == self.next_player).astype(np.int32)

        # Transposing the state for player two to be seen as win from top to bottom
        if self.current_player == PLAYERS[1]:
            player_state = player_state.T
            opponent_state = opponent_state.T

        return np.moveaxis(np.array([player_state, opponent_state]), 0, 2)

    def _on_state_updated(self):
        self._uf_state_sync()

        # Check if the last move ended the game
        last_player = self._current_player * (-1)
        self._is_game_over = self._uf[last_player].connected("start", "end")

        # If not then update the player
        if self.is_game_over:
            self.current_player = last_player
        else:
            self.current_player = last_player * (-1)

    def _state_init(self):
        shadow_state = np.zeros((self.size + 2, self.size + 2), dtype=np.int8)
        shadow_state[:, 0] = shadow_state[:, -1] = PLAYERS[1]
        shadow_state[0, :] = shadow_state[-1, :] = PLAYERS[0]
        return shadow_state

    # ####################
    #    Game Playing
    # ####################
    def play(self, move):
        if move in self.legal_moves:
            # Register the move
            self.state[move] = self.current_player

            # Sync uf neighbors
            self._uf_merge_neighbors(move, self.current_player)

            # Update game status
            self._is_game_over = self.uf.connected("start", "end")

            if not self.is_game_over:
                # Switch player
                self.current_player = self.next_player

    @property
    def legal_moves(self):
        if self.is_game_over:
            return []
        return [tuple(x) for x in np.argwhere(self.state == 0).tolist()]

    @property
    def legal_binary_moves(self):
        return np.logical_not(self.state).astype(np.int8).flatten().tolist()

    def transform_move_to_binary_move_index(self, move: tuple[int, int]) -> int:
        return move[0] * self.size + move[1]

    def transform_binary_move_index_to_move(self, binary_move_index: int) -> tuple[int, int]:
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
    def uf(self):
        return self._uf[self.current_player]

    def _uf_state_sync(self):
        # Sync with current state
        for player in PLAYERS:
            for location in np.argwhere(self.state == player).tolist():
                self._uf_merge_neighbors(location, player)

    @classmethod
    @lru_cache(typed=True)
    def _uf_neighbors(cls, location, size):
        neighbors = lambda r, c: [(r + 1, c - 1), (r, c - 1), (r - 1, c), (r - 1, c + 1), (r, c + 1), (r + 1, c)]
        inside = lambda x: 0 <= x <= size + 1
        return [(i, j) for i, j in neighbors(*location) if inside(i) and inside(j)]

    def _uf_merge_neighbors(self, location, player):
        # Merge all neighbors
        location = (location[0] + 1, location[1] + 1)
        neighbors = self._uf_neighbors(location, self.size)

        for neighbor in neighbors:
            if self._shadow_state[location] == self._shadow_state[neighbor]:
                self._uf[player].union(location, neighbor)

    def _uf_init(self):
        # Initiate union-find for both players
        uf = {
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

    def reset(self):
        self.state = np.zeros(self.state.shape, dtype=np.int8)


class HexGame(StateManager, Generic[THexGame]):
    def __init__(self, size=5, start_player=1):
        super().__init__()
        self.size = size
        self.start_player = start_player
        self.players = (1, 2)
        self.current_player, self.state, self.shadow_state, self._union_find, self._game_over = self.init_values()

    def reset(self):
        self.current_player, self.state, self.shadow_state, self._union_find, self._game_over = self.init_values()

    def init_values(self):
        state = np.zeros((self.size, self.size), dtype=np.int32)
        shadow_state = np.zeros((self.size + 2, self.size + 2), dtype=np.int32)
        shadow_state[:, 0] = shadow_state[:, -1] = self.players[0]
        shadow_state[0, :] = shadow_state[-1, :] = self.players[1]
        union_find = {
            self.players[0]: UnionFind(
                ["start", "end", *[(i, j) for i in range(0, self.size + 2) for j in range(0, self.size + 2)]]),
            self.players[1]: UnionFind(
                ["start", "end", *[(i, j) for i in range(0, self.size + 2) for j in range(0, self.size + 2)]])
        }
        for i in range(self.size + 2):
            union_find[self.players[0]].union("start", (i, self.size + 1))
            union_find[self.players[0]].union("end", (i, 0))
            union_find[self.players[1]].union("start", (0, i))
            union_find[self.players[1]].union("end", (self.size + 1, i))

        return self.start_player, state, shadow_state, union_find, False

    def sync_state(self, state):
        if state.shape == self.state.shape:
            self.state += (self.state - self.state)

    @property
    def union_find(self) -> UnionFind:
        return self._union_find[self.current_player]

    @property
    def is_game_over(self) -> bool:
        return self._game_over

    def update_game_status(self):
        self._game_over = self.union_find.connected("start", "end")

    """ MOVES """
    @property
    def legal_moves(self) -> list[tuple[int, int]]:
        if self.is_game_over:
            return []
        return [tuple(x) for x in np.argwhere(self.state == 0).tolist()]

    @property
    def legal_binary_moves(self):
        return np.logical_not(self.state).astype(np.int32).flatten().tolist()

    def transform_move_to_binary_move_index(self, move: tuple[int, int]) -> int:
        return move[0] * self.size + move[1]

    def transform_binary_move_index_to_move(self, binary_move_index: int) -> tuple[int, int]:
        return np.unravel_index(binary_move_index, shape=self.state.shape)

    @property
    def result(self):
        if self.current_player == 1:
            return 1
        else:
            return -1

    @property
    def flat_state(self):
        bits = lambda s: format(s, f"0{2}b")
        state = [self.current_player, *self.state.flatten()]
        return np.array([float(s) for s in list(''.join([bits(s) for s in state]))], dtype=np.int32)

    @property
    def cnn_state(self):
        player_state = (self.state == self.current_player).astype(np.int32)
        opponent_state = (self.state == self.next_player).astype(np.int32)

        # Transposing the state for player two to be seen as win from top to bottom
        if self.current_player == self.players[1]:
            player_state = player_state.T
            opponent_state = opponent_state.T

        return np.moveaxis(np.array([player_state, opponent_state]), 0, 2)

    @property
    def state_test(self):
        return [self.current_player, *self.state.flatten()]

    def copy(self) -> THexGame:
        return deepcopy(self)

    @property
    def next_player(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    def switch_player(self):
        self.current_player = self.next_player

    def play(self, move: tuple[int, int]):
        self.execute(move)

    def execute(self, move: tuple[int, int]):
        if move in self.legal_moves:
            shadow_move = (move[0] + 1, move[1] + 1)
            self.state[move] = self.current_player
            self.shadow_state[shadow_move] = self.current_player
            self._union_neighbors(shadow_move)

            # Cache game over
            self.update_game_status()

            if not self.is_game_over:
                self.switch_player()

    def _union_neighbors(self, move: tuple[int, int]):
        r, c = move
        neighbors = [(r+1, c-1), (r, c-1), (r-1, c), (r-1, c+1), (r, c+1), (r+1, c)]

        for neighbor in neighbors:
            if neighbor[0] < 0 or neighbor[1] < 0:
                continue
            if self.shadow_state[neighbor] == self.shadow_state[move]:
                self.union_find.union(neighbor, move)


# Classes for rendering HEX Game
class GameWindow(arcade.Window):
    def __init__(self, width, height, game: HexGame, agent: Union[Agent, None], view_update_rate=0.2):
        super().__init__(width, height, "Hex Game", update_rate=view_update_rate)
        arcade.set_background_color(arcade.color.WHITE)

        self._autoplay = False
        self.change = True
        self.game = game
        self.agent = agent
        self._agent = None
        self.renderer = StateRendering(state=self.game.state, window=(self.width, self.height))

        self.state_meta = {}
        for location in self.renderer.hexagons.keys():
            self.state_meta[location] = ""

        # Button
        self.manager = arcade.gui.UIManager()
        self.manager.enable()
        self.v_box = arcade.gui.UIBoxLayout()
        replay_button = arcade.gui.UIFlatButton(text="Reset", width=100, height=30)
        autoplay_button = arcade.gui.UIFlatButton(text="Autoplay", width=100, height=30)
        next_button = arcade.gui.UIFlatButton(text="Next", width=100, height=30)

        self.v_box.add(next_button.with_space_around(bottom=20))
        self.v_box.add(replay_button.with_space_around(bottom=20))
        self.v_box.add(autoplay_button.with_space_around(bottom=20))
        replay_button.on_click = self.reset
        autoplay_button.on_click = self.autoplay
        next_button.on_click = self.next_move

        self.manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x",
                anchor_y="center_y",
                align_y=-self.height/2+75,
                align_x=self.width/2-75,
                child=self.v_box)
        )

        self.draw_board()

    def on_draw(self):
        arcade.start_render()

    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float):
        if not self.agent:
            self.renderer.hover(x, y, self.game.current_player)
            self.draw_board()

    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int):
        if not self.agent:
            indices = self.renderer.indices_from_position(x, y)

            print("Move", indices)
            print("Before state",  self.game.state)
            self.game.execute(indices)
            print("After state", self.game.state)
            self.draw_board()

    def on_update(self, delta_time):
        if self._autoplay:
            self.next_move(None)

    def reset(self, _):
        self._autoplay = False
        self.game.reset()
        self.renderer = StateRendering(state=self.game.state, window=(self.width, self.height))
        self.state_meta = {}
        for location in self.renderer.hexagons.keys():
            self.state_meta[location] = ""
        self.agent = deepcopy(self._agent)
        self.draw_board()

    def autoplay(self, _):
        self._autoplay = not self._autoplay

    def next_move(self, _):
        if self._agent is None:
            self._agent = deepcopy(self.agent)

        if self.agent and not self.game.is_game_over:
            action, meta = self.agent.action(self.game.flat_state, self.game.legal_moves, self.game)

            if meta is not None:
                for location, value in meta.items():
                    self.state_meta[location] = value
            if action is not None:
                print(self.game.state)
                self.game.play(action)
            self.draw_board()

    def draw_board(self):
        arcade.start_render()

        # Render state meta info
        for location, hexagon in self.renderer.hexagons.items():
            arcade.draw_text(
                str(self.state_meta[location]),
                hexagon.center[0]-10,
                hexagon.center[1],
                arcade.color.GRAY,
                9,
                50,
                'left'
            )
        self.renderer.draw()
        self.manager.draw()
        arcade.draw_text('Win: ' + str(self.game.is_game_over), 20, 20, arcade.color.BLACK, 15, 180, 'left')

        if self.game.is_game_over:
            self.draw_winner_line()
        arcade.finish_render()

    def draw_winner_line(self):
        # Build a grid graph with edges in x and y direction
        graph = nx.grid_graph(dim=[self.game.size + 2, self.game.size + 2])

        # Add extra edges to be compatible with hexagons
        graph.add_edges_from(
            [
                ((x + 1, y), (x, y + 1))
                for x in range(self.game.size + 1)
                for y in range(self.game.size + 1)
            ])

        # Set all weights to 1 for all edges in the graph
        nx.set_edge_attributes(graph, 1, "weight")

        # Set boarder edges to lower weight to get correct path
        for i in range(0, self.game.size + 1):
            # BLUE
            nx.set_edge_attributes(graph, {((0, i), (0, i+1)): {"weight": 0}})
            nx.set_edge_attributes(graph, {((self.game.size + 1, i), (self.game.size + 1, i+1)): {"weight": 0}})

            # RED
            nx.set_edge_attributes(graph, {((i, 0), (i+1, 0)): {"weight": 0}})
            nx.set_edge_attributes(graph, {((i, self.game.size + 1), (i+1, self.game.size + 1)): {"weight": 0}})

        # Get the cluster with all points included in the path between start and end
        cluster = self.game.union_find.component("start")
        cluster.remove("start")
        cluster.remove("end")

        # Remove all nodes that are not part of the cluster form the graph
        for node in list(graph.nodes):
            if not (node in cluster):
                graph.remove_node(node)

        # Set the target and source node for the shortest path
        # This source node and target node is part of the border in the shadow-state
        if self.game.current_player == self.game.players[0]:
            source, target = (1, 0), (self.game.size, self.game.size + 1)
        else:
            source, target = (0, 1), (self.game.size + 1, self.game.size)

        points = nx.shortest_path(graph, source=source, target=target, weight="weight")

        # Adjust the indices from shadow state to be equivalent with the normal state
        winner_path = []
        for i, j in [(i - 1, j - 1) for i, j in points]:
            if self.game.size - 1 >= i >= 0 and self.game.size - 1 >= j >= 0:
                winner_path.append((i, j))

        # Convert indices of hexagons to center_points in window
        centers = [self.renderer.hexagons[indices].center for indices in winner_path]
        arcade.draw_line_strip(
            centers,
            line_width=list(self.renderer.hexagons.values())[0].radius * 0.5,
            color=arcade.color_from_hex_string("6afc76")
        )


def centered_zero(i):
    indexes = []
    for val in range(-math.floor(i / 2), math.floor(i / 2) + 1, 1):
        if val != 0 or (i + 2) % 2 != 0:
            indexes.append(val)
    return indexes


class Piece:
    def __init__(self, location: tuple[float, float], player: int, radius: float, active: bool = False):
        self.location = location
        self.player = player
        self.active = active
        self.r = radius

    @property
    def color(self):
        return {
            (1, True): RED_COLOR,
            (1, False): RED_COLOR_LIGHT,
            (-1, True): BLUE_COLOR,
            (-1, False): BLUE_COLOR_LIGHT,
        }[self.player, self.active]

    def draw(self):
        return arcade.draw_circle_filled(
            center_x=self.location[0],
            center_y=self.location[1],
            color=arcade.color_from_hex_string(self.color),
            radius=self.r
        )


class Hexagon:
    def __init__(self, radius: int, center: tuple[float, float]):
        self.radius = radius
        self.center = center
        self.points = self._generate_points()

        self.player = 0
        self.hover = False

    @property
    def width(self) -> float:
        return math.sqrt(3) * self.radius / 2

    @property
    def height(self):
        return self.radius

    def point_inside(self, x, y):
        n = len(self.points)
        inside = False

        p1x, p1y = self.points[0]
        for i in range(n + 1):
            p2x, p2y = self.points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def draw(self, player, hover):
        arcade.draw_polygon_outline(self.points, color=arcade.color.BLACK, line_width=2)
        if player > 0:
            piece = Piece(location=self.center, radius=self.radius-0.25*self.radius, player=player, active=(not hover))
            piece.draw()

    def _generate_points(self) -> list[tuple[float, float]]:
        points = []
        for i, j, k in [[-1, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 0], [0, 0, -1], [-1, -1, 0]]:
            points.append((self.center[0] + i*self.width, self.center[1] + j*self.height/2 + k*self.height))
        return points


class StateRendering:
    def __init__(self, state, window=(600, 750)):
        self.state = state
        self._shape = state.shape

        # Top, Bottom, Left, Right
        self._r = round((window[1] - 50)/(3*self._shape[0] - 1))
        self._start = (window[0]/2, 25)

        self.hexagons: dict[tuple[int, int], Hexagon] = {}
        self._build_matrix()

        self._hover_indices = -1, -1
        self._hover_player = None

        self._boarder_width = 4

    def indices_from_position(self, x, y):
        for k, hexagon in self.hexagons.items():
            if hexagon.point_inside(x, y):
                return k
        return -1, -1

    def draw(self):
        for k, hexagon in self.hexagons.items():
            if k == self._hover_indices and self.state[k] == 0:
                hexagon.draw(self._hover_player, True)
            else:
                hexagon.draw(self.state[k], False)

            if k[0] == 0:
                arcade.draw_line(hexagon.points[4][0], hexagon.points[4][1], hexagon.points[5][0], hexagon.points[5][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(BLUE_COLOR))
                arcade.draw_line(hexagon.points[5][0], hexagon.points[5][1], hexagon.points[0][0], hexagon.points[0][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(BLUE_COLOR))
            elif k[0] == self._shape[0] - 1:
                arcade.draw_line(hexagon.points[1][0], hexagon.points[1][1], hexagon.points[2][0], hexagon.points[2][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(BLUE_COLOR))
                arcade.draw_line(hexagon.points[2][0], hexagon.points[2][1], hexagon.points[3][0], hexagon.points[3][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(BLUE_COLOR))
            if k[1] == 0:
                arcade.draw_line(hexagon.points[3][0], hexagon.points[3][1], hexagon.points[4][0], hexagon.points[4][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(RED_COLOR))
                arcade.draw_line(hexagon.points[2][0], hexagon.points[2][1], hexagon.points[3][0], hexagon.points[3][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(RED_COLOR))
            elif k[1] == self._shape[1] - 1:
                arcade.draw_line(hexagon.points[0][0], hexagon.points[0][1], hexagon.points[1][0], hexagon.points[1][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(RED_COLOR))
                arcade.draw_line(hexagon.points[5][0], hexagon.points[5][1], hexagon.points[0][0], hexagon.points[0][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(RED_COLOR))

    def hover(self, x, y, player):
        self._hover_indices = self.indices_from_position(x, y)
        self._hover_player = player

    def _build_matrix(self):
        w = self._shape[0] + self._shape[1] - 1
        single_width = math.sqrt(3) * self._r / 2
        single_height = self._r

        for y in range(0, w):
            indices = []
            for n in range(0, self._shape[0]):
                for m in range(0, self._shape[1]):
                    if (n + m) == y:
                        indices.append((n, m))
            for i, x in enumerate(centered_zero(min(y+1, w-y))):
                center_x = (x*2 - ((y+2)%2)*int(x != 0)*math.copysign(1, x))*single_width + self._start[0]
                center_y = (y*(3/2) + 1)*single_height + self._start[1]

                self.hexagons[indices[i]] = Hexagon(
                    center=(center_x, center_y),
                    radius=self._r,
                )
