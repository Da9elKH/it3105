from environments.environment import ProblemEnvironment
from itertools import permutations
import arcade
import arcade.gui
import random
from utils.state import State, StateConstructor
from utils.types import ActionList, Action
import matplotlib.pylab as plt
import copy


class TowerOfHanoi(ProblemEnvironment):
    def __init__(self, num_pegs, num_discs, view_update_rate=0.5, time_out=300):
        super().__init__()
        self.__num_pegs = num_pegs
        self.__num_discs = num_discs
        self.__pegs = {}

        self.__episode_time_steps = {}
        self.__state_space = tuple([1]*self.__num_pegs*self.__num_discs)
        self.__state_constructor = StateConstructor(categorical_state_shape=self.__state_space)

        self.T = time_out
        self.rounds = 0
        self.__view_update_rate = view_update_rate

        assert 3 <= self.__num_pegs <= 5, "Outside pegs limitations"
        assert 2 <= self.__num_discs <= 6, "Outside discs limitations"

    def input_space(self) -> int:
        """ Input space is used to initialize the first input layer of NN"""
        return len(self.__state_constructor(self.__state_space).array)

    def __action_space(self) -> list[tuple]:
        """ Get the available actions for this game"""
        return list(permutations(range(self.__num_pegs), 2))

    def legal_actions(self) -> ActionList:
        """ Defines the legal actions in current state"""
        legal_actions = []
        for from_peg, to_peg in self.__action_space():
            if not self.__pegs[from_peg]:
                continue
            elif not self.__pegs[to_peg]:
                legal_actions.append((from_peg, to_peg))
            elif self.__pegs[from_peg][-1] < self.__pegs[to_peg][-1]:
                legal_actions.append((from_peg, to_peg))

        return legal_actions

    def step(self, action: Action) -> tuple[State, Action, float, State, ActionList, bool]:
        """ Run the next action in the given environment """
        from_state = self.state()
        from_peg = action[0]
        to_peg = action[1]

        self.__pegs[to_peg].append(self.__pegs[from_peg].pop())
        self.rounds += 1

        return from_state, action, self.reinforcement(), self.state(), self.legal_actions(), self.__in_terminal_state()

    def state(self) -> State:
        """ Returns a tuple containing the state of the game """
        result = [0]*self.__num_pegs*self.__num_discs
        for peg, discs in self.__pegs.items():
            for disc in discs:
                result[(peg - 1)*self.__num_discs + (disc - 1)] = 1
        return self.__state_constructor(tuple(result))

    def reinforcement(self) -> float:
        """ By giving negative reward we are optimizing for fewer moves """
        if self.__has_succeeded():
            return 1.0
        else:
            return -1.0

    def __has_failed(self) -> bool:
        """ Check if problem failed (in a terminal state) """
        return False

    def __has_succeeded(self):
        """ Check if problem is finished (in a terminal state) """
        for discs in [len(i) for i in list(self.__pegs.values())[1:]]:
            if discs == self.__num_discs:
                return True
        return False

    def __in_terminal_state(self) -> bool:
        """ Check if problem is in a terminal state (used for bootstrapping)"""
        return self.__has_succeeded() or self.__has_failed()

    def __has_timed_out(self) -> bool:
        """ Check if problem has timed out (used to e"""
        if self.rounds >= self.T:
            return True
        return False

    def is_finished(self) -> bool:
        """ Check if the episode is finished"""
        return self.__in_terminal_state() or self.__has_timed_out()

    def reset(self) -> tuple[State, ActionList]:
        """ Resets the environment parameters to initial state before next episode """
        self.__pegs.clear()
        self.rounds = 0

        # Set all discs at the first peg in state
        for i in range(self.__num_pegs):
            self.__pegs[i] = []
        self.__pegs[0] = list(range(self.__num_discs, 0, -1))

        return self.state(), self.legal_actions()

    def store_training_metadata(self, last_episode, current_episode, current_step, state, reinforcement):
        """ This function stores metadata for later replay of the episode """
        self.__episode_time_steps[current_episode] = current_step
        if last_episode:
            self.replay_states.append(state)

    def replay(self, saps, values):
        """ This function runs at the end of the training for visualization """
        x, y = zip(*self.__episode_time_steps.items())
        plt.plot(x, y)
        plt.show()

        render = RenderWindow(600, 480, "Tower of Hanoi", self.__num_pegs, self.__num_discs, self.replay_states, self.__view_update_rate)
        render.setup()
        arcade.run()


""" RENDERING OF TOWER OF HANOI"""
class RenderWindow(arcade.Window):
    def __init__(self, width, height, title, num_pegs, num_discs, states, view_update_rate):
        super().__init__(width, height, title, update_rate=view_update_rate)
        self.__num_pegs = num_pegs
        self.__num_discs = num_discs

        self.__disc_objects = {}
        self.__pegs_objects = {}

        self.__pegs_list = arcade.SpriteList()
        self.__disc_list = arcade.SpriteList()

        self.__states = states[::-1]
        self.states = states[::-1]
        self.moves = ""

        self.manager = arcade.gui.UIManager()
        self.manager.enable()
        self.v_box = arcade.gui.UIBoxLayout()

        arcade.set_background_color(arcade.color.WHITE)

        replay_button = arcade.gui.UIFlatButton(text="Replay", width=100, height=40)
        self.v_box.add(replay_button.with_space_around(bottom=20))

        replay_button.on_click = self.on_click_start

        self.manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x",
                anchor_y="center_y",
                align_y=125,
                child=self.v_box)
        )

    def on_draw(self):
        arcade.start_render()
        arcade.draw_line(0, 50, self.width, 50, arcade.color.GRAY, line_width=5)

        self.manager.draw()
        arcade.draw_text('Moves: ' + str(self.moves), self.width/2 - 180/2, self.height - 50, arcade.color.BLACK, 20, 180, 'center')

        self.__pegs_list.draw()
        self.__disc_list.draw()

    def on_click_start(self, event):
        self.states = copy.deepcopy(self.__states)
        self.moves = ""

    def on_update(self, delta_time):
        # Render Tower of Hanoi based on State
        if self.states:
            if self.moves == "":
                self.moves = 0
            else:
                self.moves += 1

            state: State = self.states.pop()
            discs = dict([(i, []) for i in range(self.__num_pegs)])

            for i, val in enumerate(state.array):
                if not bool(val):
                    continue

                disc = i%self.__num_discs
                peg = i/self.__num_discs - disc/self.__num_discs
                discs[round(peg)].append((disc, self.__disc_objects[round(disc)]))

            for k in range(self.__num_pegs):
                self.__pegs_objects[k].move_discs(dict(discs[k]))

        self.__disc_list.update()

    def setup(self):
        pegs_offset = 50
        pegs_space = round((self.width - pegs_offset * 2) / (self.__num_pegs + 1))
        pegs_width = 10
        pegs_height = 200

        disc_height = 20
        disc_min_width = 30
        disc_max_width = min(100, pegs_space)
        disc_increment = (disc_max_width - disc_min_width) / self.__num_discs

        # Generate pegs and disc objects
        for i in range(self.__num_discs):
            self.__disc_objects[i] = Disc(round(disc_min_width + disc_increment * i), disc_height)

        for i in range(self.__num_pegs):
            self.__pegs_objects[i] = Peg(pegs_offset + (i + 1) * pegs_space, 52.5 + pegs_height / 2, pegs_width, pegs_height)

        # Add objects to sprite-lists
        for peg in self.__pegs_objects.values():
            self.__pegs_list.append(peg)
        for disc in self.__disc_objects.values():
            self.__disc_list.append(disc)

class Disc(arcade.SpriteSolidColor):
    def __init__(self, width, height):
        color_hex = "#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])
        color = arcade.color_from_hex_string(color_hex)
        super().__init__(width, height, color)

class Peg(arcade.SpriteSolidColor):
    def __init__(self, x, y, width, height):
        super().__init__(width, height, color=arcade.color.BROWN_NOSE)
        self.center_x = x
        self.center_y = y

    def move_discs(self, discs):
        offset_y = self.center_y - self.height/2
        sort = sorted(discs, reverse=True)
        for k in sort:
            discs[k].center_y = offset_y + discs[k].height/2
            discs[k].center_x = self.center_x
            offset_y += discs[k].height
