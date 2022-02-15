import math
import arcade

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

class GameEnvironment:
    def action_space(self) -> list[int]:
        """ Get the available actions for this game"""
        pass

    def step(self, action: int) -> list[any]:
        """ Run the next """
        pass

    def has_failed(self) -> bool:
        """ Check if agent failed """
        pass


class CartPolePhysics(GameEnvironment):
    def __init__(self, theta):
        self.F = 10

        self.L = 0.5  # Length of pole
        self.mp = 0.1  # Mass of pole
        self.mc = 1  # Mass of cart
        self.g = 9.8  # Gravity
        self.theta = theta  # Angle of pole (radians)
        self.dtheta = 0  # First temporal derivative of the pole angle
        self.ddtheta = 0  # Second temporal derivative of the pole angle
        self.x = 0  # Horizontal location of cart
        self.dx = 0  # Horizontal velocity of cart
        self.ddx = 0  # Horizontal acceleration of cart
        self.t = 0.02  # The timestep for one simulation
        self.T = 300  # Number of timesteps in one simulation

        self.viewer = None
        self.state = None

    def has_failed(self) -> bool:
        # Check if pole i outside of limits
        # Check if cart is outside of limits
        return False

    def action_space(self) -> list[int]:
        """
            0: Force left
            1: Force right
        """
        return [0, 1]

    def step(self, action: int) -> list[any]:
        # Calculate the direction and size of the force
        B = (self.F * action) or (-self.F)

        # Calculate new angle for pole
        self.ddtheta = (
            (self.g*math.sin(self.theta)+(math.cos(self.theta)*(-B-self.mp*self.L*(self.dtheta**2)*math.sin(self.theta)))/(self.mc*self.mp))
            /
            (self.L*((4/3)-((self.mp*(math.cos(self.theta)**2))/(self.mp+self.mc))))
        )

        # Calculate new location of
        self.ddx = (
            (B+(self.mp*self.L*((self.dtheta**2)*math.sin(self.theta)-self.ddtheta*math.cos(self.theta))))
            /
            (self.mp+self.mc)
        )

        self.dtheta = self.dtheta + self.t*self.ddtheta
        self.theta = self.theta + self.t*self.dtheta

        self.dx = self.dx + self.t*self.ddx
        self.x = self.x + self.t*self.dx

        self.state = [self.x, self.dx, self.theta, self.dtheta]
        return self.state


class Cart(arcade.SpriteSolidColor):
    def __init__(self, x, y, width, height, color, ):
        super().__init__(width, height, color)
        self.center_x = x
        self.center_y = y

class Pole(arcade.SpriteSolidColor):
    def __init__(self, x, y, width, height, connection_y, theta, color):
        super().__init__(width, height, color)
        self.connection_y = connection_y
        self.center_x = x
        self.center_y = y

        self.change_angle = theta
        self.update()

    def update(self):
        if self.change_angle:
            self.angle = math.degrees(-self.change_angle)
            self.change_angle = None

        self.adjust_pole_to_connection()
        self.position = [
            self._position[0] + self.change_x,
            self._position[1] + self.change_y,
        ]

    def adjust_pole_to_connection(self):
        # Calculate the offset after rotation
        edges = self.get_adjusted_hit_box()
        bottom_y = (edges[0][1] + edges[1][1]) / 2
        bottom_x = (edges[0][0] + edges[1][0]) / 2

        # Adjust the change in y and x based on the offset
        self.change_y = (self.connection_y - bottom_y)
        self.change_x = (self.center_x - bottom_x)

class PoleConnection(arcade.SpriteCircle):
    def __init__(self, x, y, radius, color):
        super().__init__(radius, color)
        self.center_x = x
        self.center_y = y

class CartPoleCombined:
    def __init__(self, theta):
        self.offset_x = SCREEN_WIDTH/2
        self.pole_y = 50

        self.env = CartPolePhysics(theta=theta)
        self.cart = Cart(self.offset_x, 25, 100, 50, arcade.color.LIGHT_GRAY)
        self.pole = Pole(self.offset_x, 100, 10, 100, self.pole_y, theta, arcade.color.LIGHT_BROWN)
        self.pole_connection = PoleConnection(self.offset_x, 50, 5, arcade.color.BLACK)

    def draw(self):
        self.cart.draw()
        self.pole.draw()
        self.pole_connection.draw()

    def move(self, right=True):
        if right:
            x, dx, theta, dtheta = self.env.step(1)
        else:
            x, dx, theta, dtheta = self.env.step(0)

        print(x)

        # Scale this for easier movements
        x = (x*300 + self.offset_x)

        self.cart.set_position(center_x=x, center_y=self.cart.center_y)
        self.pole_connection.set_position(center_x=x, center_y=self.pole_connection.center_y)
        self.pole.set_position(center_x=x, center_y=self.pole.center_y)
        self.pole.change_angle = theta
        self.pole.update()

class Rendering(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        self.set_mouse_visible(False)
        arcade.set_background_color(arcade.color.WHITE)

        # Objects
        self.player = CartPoleCombined(theta=-0.21)

    def on_draw(self):
        arcade.start_render()
        self.player.draw()

        # Draw the ground
        arcade.draw_line(0, 0, SCREEN_WIDTH, 0, arcade.color.BLACK)

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.RIGHT:
            self.player.move(right=True)
        elif symbol == arcade.key.LEFT:
            self.player.move(right=False)
