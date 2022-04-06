from typing import Union, Dict, Tuple, List
import arcade
import arcade.gui
import networkx as nx
import math
from agents import Agent
from copy import deepcopy
from environments import HexGame, PLAYERS
from misc import Move
from typing import Callable
import numpy as np

RED_COLOR = "fc766a"
RED_COLOR_LIGHT = "fed1cd"
BLUE_COLOR = "5b84b1"
BLUE_COLOR_LIGHT = "bacbde"


class HexGUI(arcade.Window):
    def __init__(self, environment=HexGame(size=7), agent: Union[Agent, None] = None, width=1000, height=700, view_update_rate=None):
        if agent:
            view_update_rate = 0.2

        super().__init__(width, height, "Hex Game", update_rate=view_update_rate)
        arcade.set_background_color(arcade.color.WHITE)

        self._autoplay = False
        self.change = True
        self.environment = environment
        self.agent = agent
        self._agent = None
        self.renderer = StateRendering(state=self.environment.state, window=(self.width, self.height))

        self.state_meta = {}
        for location in self.renderer.hexagons.keys():
            self.state_meta[location] = ""

        # Buttons
        self.manager = arcade.gui.UIManager()
        self.manager.enable()
        self.v_box = arcade.gui.UIBoxLayout()

        reset_button = arcade.gui.UIFlatButton(text="Reset", width=100, height=30)
        next_button = arcade.gui.UIFlatButton(text="Next", width=100, height=30)
        autoplay_button = arcade.gui.UIFlatButton(text="Autoplay", width=100, height=30)

        self.v_box.add(reset_button.with_space_around(bottom=20))
        self.v_box.add(next_button.with_space_around(bottom=20))
        self.v_box.add(autoplay_button.with_space_around(bottom=20))

        reset_button.on_click = self.reset
        next_button.on_click = self.next_move
        autoplay_button.on_click = self.autoplay

        self.manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x",
                anchor_y="center_y",
                align_y=-self.height / 2 + 75,
                align_x=self.width / 2 - 75,
                child=self.v_box)
        )
        self.draw_board()
        self.visualization_loop = None

        # Register hooks
        self.environment.register_move_hook(lambda move: self.draw_board())
        self.environment.register_reset_hook(lambda: self.draw_board())

    def run_visualization_loop(self, function: Callable):
        self.visualization_loop = function

        # Change the default menu to fit that it is inside a visualization loop
        unused_buttons = []
        for element in self.v_box.children:
            if element.children[0].text == "Autoplay":
                element.children[0].text = "Run"
            else:
                unused_buttons.append(element)
        for element in unused_buttons:
            self.v_box.remove(element)

        self.manager.children[0][0].align_y = -self.height / 2 + 25*len(self.v_box.children)
        self.draw_board()
        self.run()

    def on_draw(self):
        arcade.start_render()

    def register_move_on_board(self, move: Move):
        if not self.environment.is_game_over:
            self.environment.play(move)

    def hover_move_on_board(self, x: float, y: float):
        self.renderer.hover(x, y, self.environment.current_player)
        self.draw_board()

    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float):
        self.hover_move_on_board(x, y)

    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int):
        indices = self.renderer.indices_from_position(x, y)
        self.register_move_on_board(indices)

    def on_update(self, delta_time):
        if self._autoplay:
            self.next_move(None)

    def reset(self, _):
        self._autoplay = False
        self.state_meta = {location: "" for location in self.renderer.hexagons.keys()}
        self.environment.reset()
        self.draw_board()

    def autoplay(self, _):
        self._autoplay = not self._autoplay
        if self.visualization_loop:
            self.visualization_loop()

    def next_move(self, _):
        if self._agent is None:
            self._agent = deepcopy(self.agent)

        if self.agent and not self.environment.is_game_over and self.environment.current_player == -1:
            action, dist = self.agent.get_move(greedy=True)
            if dist is not None:
                size = (self.environment.size, self.environment.size)
                dist = np.array(dist).reshape((self.environment.size, self.environment.size))
                for location in list(np.ndindex(size)):
                    self.state_meta[location] = str(round(dist[location], 2))
            if action is not None:
                self.environment.play(action)
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

        if self.environment.is_game_over:
            self.draw_winner_line()

        arcade.finish_render()

    def draw_winner_line(self):
        # Build a grid graph with edges in x and y direction
        graph = nx.grid_graph(dim=[self.environment.size + 2, self.environment.size + 2])

        # Add extra edges to be compatible with hexagons
        graph.add_edges_from(
            [
                ((x + 1, y), (x, y + 1))
                for x in range(self.environment.size + 1)
                for y in range(self.environment.size + 1)
            ])

        # Set all weights to 1 for all edges in the graph
        nx.set_edge_attributes(graph, 1, "weight")

        # Set boarder edges to lower weight to get correct path
        for i in range(0, self.environment.size + 1):
            # BLUE
            nx.set_edge_attributes(graph, {((0, i), (0, i + 1)): {"weight": 0}})
            nx.set_edge_attributes(graph, {((self.environment.size + 1, i), (self.environment.size + 1, i + 1)): {"weight": 0}})

            # RED
            nx.set_edge_attributes(graph, {((i, 0), (i + 1, 0)): {"weight": 0}})
            nx.set_edge_attributes(graph, {((i, self.environment.size + 1), (i + 1, self.environment.size + 1)): {"weight": 0}})

        # Get the cluster with all points included in the path between start and end
        cluster = self.environment.uf.component("start")

        cluster.remove("start")
        cluster.remove("end")

        # Remove all nodes that are not part of the cluster form the graph
        for node in list(graph.nodes):
            if not (node in cluster):
                graph.remove_node(node)

        # Set the target and source node for the shortest path
        # This source node and target node is part of the border in the shadow-state
        if self.environment.current_player == PLAYERS[1]:
            source, target = (1, 0), (self.environment.size, self.environment.size + 1)
        else:
            source, target = (0, 1), (self.environment.size + 1, self.environment.size)

        points = nx.shortest_path(graph, source=source, target=target, weight="weight")

        # Adjust the indices from shadow state to be equivalent with the normal state
        winner_path = []
        for i, j in [(i - 1, j - 1) for i, j in points]:
            if self.environment.size - 1 >= i >= 0 and self.environment.size - 1 >= j >= 0:
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
    def __init__(self, location: Tuple[float, float], player: int, radius: float, active: bool = False):
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
    def __init__(self, radius: int, center: Tuple[float, float]):
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
        if player in PLAYERS:
            piece = Piece(location=self.center, radius=self.radius - 0.25 * self.radius, player=player,
                          active=(not hover))
            piece.draw()

    def _generate_points(self) -> List[Tuple[float, float]]:
        points = []
        for i, j, k in [[-1, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 0], [0, 0, -1], [-1, -1, 0]]:
            points.append((self.center[0] + i * self.width, self.center[1] + j * self.height / 2 + k * self.height))
        return points


class StateRendering:
    def __init__(self, state, window=(600, 750)):
        self.state = state
        self._shape = state.shape

        # Top, Bottom, Left, Right
        self._r = round((window[1] - 50) / (3 * self._shape[0] - 1))
        self._start = (window[0] / 2, 25)

        self.hexagons: Dict[Tuple[int, int], Hexagon] = {}
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
                                 color=arcade.color_from_hex_string(RED_COLOR))
                arcade.draw_line(hexagon.points[5][0], hexagon.points[5][1], hexagon.points[0][0], hexagon.points[0][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(RED_COLOR))
            elif k[0] == self._shape[0] - 1:
                arcade.draw_line(hexagon.points[1][0], hexagon.points[1][1], hexagon.points[2][0], hexagon.points[2][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(RED_COLOR))
                arcade.draw_line(hexagon.points[2][0], hexagon.points[2][1], hexagon.points[3][0], hexagon.points[3][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(RED_COLOR))
            if k[1] == 0:
                arcade.draw_line(hexagon.points[3][0], hexagon.points[3][1], hexagon.points[4][0], hexagon.points[4][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(BLUE_COLOR))
                arcade.draw_line(hexagon.points[2][0], hexagon.points[2][1], hexagon.points[3][0], hexagon.points[3][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(BLUE_COLOR))
            elif k[1] == self._shape[1] - 1:
                arcade.draw_line(hexagon.points[0][0], hexagon.points[0][1], hexagon.points[1][0], hexagon.points[1][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(BLUE_COLOR))
                arcade.draw_line(hexagon.points[5][0], hexagon.points[5][1], hexagon.points[0][0], hexagon.points[0][1],
                                 line_width=self._boarder_width,
                                 color=arcade.color_from_hex_string(BLUE_COLOR))

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
            for i, x in enumerate(centered_zero(min(y + 1, w - y))):
                center_x = (x * 2 - ((y + 2) % 2) * int(x != 0) * math.copysign(1, x)) * single_width + self._start[0]
                center_y = (y * (3 / 2) + 1) * single_height + self._start[1]

                self.hexagons[indices[i]] = Hexagon(
                    center=(center_x, center_y),
                    radius=self._r,
                )

# TODO: REMOVE THIS (?)
if __name__ == "__main__":
    from agents import CNNAgent
    from networks import CNN

    env = HexGame(size=7)
    gui = HexGUI(environment=env, agent=CNNAgent(environment=env, network=CNN.from_file("7x7/(1) CNN_S7_B1533.h5")))
    gui.run()
