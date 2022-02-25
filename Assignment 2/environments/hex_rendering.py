import math
import arcade

RED_COLOR = "fc766a"
RED_COLOR_LIGHT = "fed1cd"
BLUE_COLOR = "5b84b1"
BLUE_COLOR_LIGHT = "bacbde"


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
            (2, True): BLUE_COLOR,
            (2, False): BLUE_COLOR_LIGHT,
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

        """
        centers = []
        for k, hexagon in self.hexagons.items():
            if self.state[k] > 0:
                centers.append(hexagon.center)

        if len(centers) > 1:
            for i in range(0, len(centers) - 1):
                arcade.draw_line(centers[i][0], centers[i][1], centers[i+1][0], centers[i+1][1],
                                 line_width=5,
                                 color=arcade.color.GREEN)
        """

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
