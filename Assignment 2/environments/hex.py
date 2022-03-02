import arcade
import arcade.gui
import numpy as np
import networkx as nx
import random
from unionfind import UnionFind
from .hex_rendering import StateRendering


class HexGame:
    def __init__(self, size=5):
        self.size = size
        self.player = 1

        # Generate states (shadow=for winning check, state=for pieces placement)
        self.state = np.zeros((self.size, self.size), dtype=np.int32)
        self.shadow_state = np.zeros((self.size + 2, self.size + 2), dtype=np.int32)
        self.shadow_state[:, 0] = self.shadow_state[:, -1] = 1
        self.shadow_state[0, :] = self.shadow_state[-1, :] = 2
        self.available_spots = set([(i, j) for i in range(0, self.size) for j in range(0, self.size)])

        # UnionFind for win checking
        self._union_find = None
        self.setup_union_find()

    def reset(self):
        self.player = 1
        self.state = np.zeros((self.size, self.size), dtype=np.int32)
        self.shadow_state = np.zeros((self.size + 2, self.size + 2), dtype=np.int32)
        self.shadow_state[:, 0] = self.shadow_state[:, -1] = 1
        self.shadow_state[0, :] = self.shadow_state[-1, :] = 2
        self.available_spots = set([(i, j) for i in range(0, self.size) for j in range(0, self.size)])
        self.setup_union_find()

    def setup_union_find(self):
        self._union_find = [
            UnionFind(["start", "end", *[(i, j) for i in range(0, self.size + 2) for j in range(0, self.size + 2)]]),
            UnionFind(["start", "end", *[(i, j) for i in range(0, self.size + 2) for j in range(0, self.size + 2)]])
        ]

        for i in range(self.size + 2):
            self._union_find[0].union("start", (i, self.size + 1))
            self._union_find[0].union("end", (i, 0))
            self._union_find[1].union("start", (0, i))
            self._union_find[1].union("end", (self.size + 1, i))

    @property
    def union_find(self) -> UnionFind:
        return self._union_find[self.player - 1]

    @property
    def winner(self):
        return self.union_find.connected("start", "end")

    @property
    def legal_moves(self):
        if not self.winner:
            return list(self.available_spots)
        else:
            return []

    @property
    def result(self):
        if self.player == 1:
            return 1
        else:
            return -1

    @property
    def flat_state(self):
        return [self.player, *self.state.flatten()]

    def switch_player(self):
        self.player = 1 if self.player == 2 else 2

    def step(self, indices):
        if indices in self.legal_moves:
            shadow_indices = (indices[0] + 1, indices[1] + 1)
            self.available_spots.remove(indices)
            self.state[indices] = self.player
            self.shadow_state[shadow_indices] = self.player
            self.register_neighbors(shadow_indices)

            if not self.winner:
                self.switch_player()

    def register_neighbors(self, indices: tuple[int, int]):
        r, c = indices
        neighbor_indices = [(r+1, c-1), (r, c-1), (r-1, c), (r-1, c+1), (r, c+1), (r+1, c)]

        for n_indices in neighbor_indices:
            if n_indices[0] >= 0 and n_indices[1] >= 0:
                if self.shadow_state[n_indices] == self.shadow_state[indices]:
                    self.union_find.union(n_indices, indices)


class GameWindow(arcade.Window):
    def __init__(self, width, height, view_update_rate=0.2):
        super().__init__(width, height, "Hex Game", update_rate=view_update_rate)
        arcade.set_background_color(arcade.color.WHITE)

        self.renderer = None
        self.engine = None
        self.change = True
        self.agent = None

        arcade.set_background_color(arcade.color.WHITE)

        # Button
        self.manager = arcade.gui.UIManager()
        self.manager.enable()
        self.v_box = arcade.gui.UIBoxLayout()

        replay_button = arcade.gui.UIFlatButton(text="Reset", width=100, height=30)
        self.v_box.add(replay_button.with_space_around(bottom=20))

        replay_button.on_click = self.reset

        self.manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x",
                anchor_y="center_y",
                align_y=-self.height/2+20,
                align_x=self.width/2-75,
                child=self.v_box)
        )

    def setup(self, engine, agent):
        self.engine = engine
        self.renderer: StateRendering = StateRendering(
            state=self.engine.state,
            window=(self.width, self.height)
        )
        self.agent = agent
        self.draw_board()

    def on_draw(self):
        arcade.start_render()

    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float):
        self.renderer.hover(x, y, self.engine.player)
        self.draw_board()

    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int):
        indices = self.renderer.indices_from_position(x, y)
        self.engine.step(indices)
        self.draw_board()

    def on_update(self, delta_time):
        if self.agent and not self.engine.winner:
            action = self.agent.action(self.engine.flat_state, self.engine.legal_moves)
            self.engine.step(action)
            self.draw_board()

    def reset(self, event):
        self.engine.reset()
        self.renderer = StateRendering(
            state=self.engine.state,
            window=(self.width, self.height)
        )
        self.draw_board()

    def draw_board(self):
        arcade.start_render()
        self.renderer.draw()
        self.manager.draw()
        arcade.draw_text('Win: ' + str(self.engine.winner), 20, 20, arcade.color.BLACK, 15, 180, 'left')
        if self.engine.winner:
            self.draw_winner_line()
        arcade.finish_render()

    def draw_winner_line(self):
        # Build a grid graph with edges in x and y direction
        graph = nx.grid_graph(dim=[self.engine.size + 2, self.engine.size + 2])

        # Add extra edges to be compatible with hexagons
        graph.add_edges_from(
            [
                ((x + 1, y), (x, y + 1))
                for x in range(self.engine.size + 1)
                for y in range(self.engine.size + 1)
            ])

        # Set all weights to 1 for all edges in the graph
        nx.set_edge_attributes(graph, 1, "weight")

        # Set boarder edges to lower weight to get correct path
        for i in range(0, self.engine.size + 1):
            # BLUE
            nx.set_edge_attributes(graph, {((0, i), (0, i+1)): {"weight": 0}})
            nx.set_edge_attributes(graph, {((self.engine.size + 1, i), (self.engine.size + 1, i+1)): {"weight": 0}})

            # RED
            nx.set_edge_attributes(graph, {((i, 0), (i+1, 0)): {"weight": 0}})
            nx.set_edge_attributes(graph, {((i, self.engine.size + 1), (i+1, self.engine.size + 1)): {"weight": 0}})

        # Get the cluster with all points included in the path between start and end
        cluster = self.engine.union_find.component("start")
        cluster.remove("start")
        cluster.remove("end")

        # Remove all nodes that are not part of the cluster form the graph
        for node in list(graph.nodes):
            if not (node in cluster):
                graph.remove_node(node)

        # Set the target and source node for the shortest path
        # This source node and target node is part of the border in the shadow-state
        if self.engine.player == 1:
            source, target = (1, 0), (self.engine.size, self.engine.size + 1)
        else:
            source, target = (0, 1), (self.engine.size + 1, self.engine.size)

        points = nx.shortest_path(graph, source=source, target=target, weight="weight")

        # Adjust the indices from shadow state to be equivalent with the normal state
        winner_path = []
        for i, j in [(i - 1, j - 1) for i, j in points]:
            if self.engine.size - 1 >= i >= 0 and self.engine.size - 1 >= j >= 0:
                winner_path.append((i, j))

        # Convert indices of hexagons to center_points in window
        centers = [self.renderer.hexagons[indices].center for indices in winner_path]
        arcade.draw_line_strip(
            centers,
            line_width=list(self.renderer.hexagons.values())[0].radius * 0.5,
            color=arcade.color_from_hex_string("6afc76")
        )