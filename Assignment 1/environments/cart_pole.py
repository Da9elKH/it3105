from environments.environment import ProblemEnvironment
from utils.state import StateConstructor, State
from utils.types import Action, ActionList
import random
import math
import arcade
import matplotlib.pylab as plt
from scipy.stats import binned_statistic


class CartPole(ProblemEnvironment):
    def __init__(self, pole_length=0.5, pole_mass=0.1, gravity=-9.8, time_step=0.02, buckets=None, time_out=300, view_update_rate=None):
        super().__init__()

        self.__theta_limits = [-0.21, 0.21]
        self.__x_limits = [-2.4, 2.4]

        self.F = 10
        self.T = time_out  # Number of time steps in one simulation
        self.mc = 1  # Mass of cart
        self.mp = pole_mass  # Mass of pole
        self.mt = self.mp + self.mc
        self.g = gravity  # Gravity
        self.t = time_step  # The time step for one simulation
        self.L = pole_length  # Length of pole

        self.__episode_time_steps = {}
        self.data = None
        self.rounds = 0
        self.__view_update_rate = view_update_rate or time_step

        self.__state_shape = buckets
        self.__state_constructor = StateConstructor(
            categorical_state_shape=self.__state_shape,
            binary_array=False
        )
        self.reset()

    """ STATE HANDLING """
    def reset(self) -> tuple[State, ActionList]:
        """ Reset the parameters before next episode """
        self.data = (0, 0, self.random_theta(), 0)
        self.rounds = 0
        return self.state(), self.legal_actions()

    def random_theta(self) -> float:
        """ Calculate a random theta """
        e = 0.1 ** 10  # Small number to avoid starting in a terminal state
        return random.uniform(self.__theta_limits[0] + e, self.__theta_limits[1] - e)

    def state(self) -> State:
        """ Return the state representation """
        return self.__state_constructor(self.__bins(), self.data)

    def __bins(self) -> tuple[int]:
        """ Function for bucketizing the state """
        low = [self.__x_limits[0], -2, self.__theta_limits[0], -2]
        high = [self.__x_limits[1], 2, self.__theta_limits[1], 2]

        bin_state = tuple(
            [
                binned_statistic(
                    max(low[i], min(high[i], self.data[i])),
                    max(low[i], min(high[i], self.data[i])),
                    bins=self.__state_shape[i],
                    range=(low[i], high[i])
                )[2][0] for i in range(len(self.data))
            ]
        )

        return bin_state

    """ STATE SPACE """
    def input_space(self):
        """ Defining the input space for this environment (used for NN) """
        return len(self.__state_constructor(self.__state_shape, (0, 0, 0, 0)).array)

    def __action_space(self) -> list[int]:
        return [0, 1]

    def legal_actions(self) -> list[int]:
        return self.__action_space()

    """ ENVIRONMENT EXECUTION """
    def step(self, action: Action) -> tuple[State, Action, float, State, ActionList, bool]:
        """ Execute a step in the environment """
        assert self.data is not None, "You must call reset() before running simulation"

        # This is different because the state is "rounded" version
        x, dx, theta, dtheta = self.data
        from_state = self.state()

        # Calculate the direction and size of the force
        B = self.F if action == 1 else -self.F

        # Calculate new values for x, dx, theta and dtheta
        sintheta = math.sin(theta)
        costheta = math.cos(theta)

        # Accelerations
        temp = (B + self.mp*self.L * dtheta**2 * sintheta)/self.mt
        ddtheta = (self.g*sintheta-costheta*temp)/(self.L*(4.0/3.0 - (self.mp * costheta**2 / self.mt)))
        ddx = temp - (self.mp * self.L * ddtheta * costheta)/self.mt

        # Update the state values
        dtheta = dtheta + self.t*ddtheta
        dx = dx + self.t*ddx
        theta = theta + self.t*dtheta
        x = x + self.t*dx

        # Store data about this step
        self.rounds += 1
        self.data = (x, dx, theta, dtheta)

        return from_state, action, self.reinforcement(), self.state(), self.legal_actions(), self.__in_terminal_state()

    def reinforcement(self) -> float:
        if self.__has_failed():
            return 0
        return 1.0

    """ STATE STATUS """
    def __has_succeeded(self) -> bool:
        return False

    def __has_failed(self) -> bool:
        x, _, theta, _ = self.data

        if not (self.__theta_limits[0] <= theta <= self.__theta_limits[1]):
            return True
        elif not (self.__x_limits[0] <= x <= self.__x_limits[1]):
            return True
        else:
            return False

    def __in_terminal_state(self) -> bool:
        return self.__has_succeeded() or self.__has_failed()

    def __has_timed_out(self) -> bool:
        if self.rounds >= self.T:
            return True
        return False

    def is_finished(self) -> bool:
        return self.__in_terminal_state() or self.__has_timed_out()

    """ REPLAY """
    def store_training_metadata(self, last_episode, current_episode, current_step, state, reinforcement):
        """ Stores data for later replay """
        self.__episode_time_steps[current_episode] = current_step
        if last_episode:
            self.replay_states.append(self.data)

    def replay(self, saps, values):
        """ Replay after the last episode """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        last_episode = {}
        last_episode_x = {}
        for i in range(len(self.replay_states)):
            last_episode[i] = self.replay_states[i][2]
            last_episode_x[i] = self.replay_states[i][1]

        x, y = zip(*last_episode.items())
        ax1.plot(x, y)

        x, y = zip(*last_episode_x.items())
        ax2.plot(x, y)

        x, y = zip(*self.__episode_time_steps.items())
        ax3.plot(x, y)
        plt.show()

        RenderWindow(600, 480, "Cart Pole Balancing", self.replay_states, self.__view_update_rate)
        arcade.run()


""" RENDER CLASSES """
class RenderWindow(arcade.Window):
    def __init__(self, width, height, title, states, view_update_rate):
        super().__init__(width, height, title, update_rate=view_update_rate)
        arcade.set_background_color(arcade.color.WHITE)

        # Objects
        self.states = states[::-1]
        first_state = self.states.pop()
        self.player = CartPoleCombined(theta=first_state[2], offset_x=self.width/2)

    def on_draw(self):
        arcade.start_render()
        self.player.draw()

        # Draw the ground
        arcade.draw_line(0, 0, self.width, 0, arcade.color.BLACK)

    def on_update(self, delta_time):
        if self.states:
            state = self.states.pop()
            x = state[0]
            theta = state[2]
            self.player.move(x, theta)


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
    def __init__(self, theta, offset_x):
        self.offset_x = offset_x
        self.pole_y = 50

        self.cart = Cart(self.offset_x, 25, 100, 50, arcade.color.LIGHT_GRAY)
        self.pole = Pole(self.offset_x, 100, 10, 100, self.pole_y, theta, arcade.color.LIGHT_BROWN)
        self.pole_connection = PoleConnection(self.offset_x, 50, 5, arcade.color.BLACK)

    def draw(self):
        self.cart.draw()
        self.pole.draw()
        self.pole_connection.draw()

    def move(self, x, theta):
        # Scale this for easier movements
        x = (x * 30 + self.offset_x)

        self.cart.set_position(center_x=x, center_y=self.cart.center_y)
        self.pole_connection.set_position(center_x=x, center_y=self.pole_connection.center_y)
        self.pole.set_position(center_x=x, center_y=self.pole.center_y)
        self.pole.change_angle = theta
        self.pole.update()

