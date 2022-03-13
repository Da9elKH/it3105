from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import numpy as np


class Actor:
    def __init__(self, input_size: int, output_size: int, hidden_size: tuple[int, ...], learning_rate: float):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._learning_rate = learning_rate
        self.nn = self.build_network()

    def build_network(self):
        """ Initialize the NN with the given depth and width for the problem environment """
        model = Sequential()
        model.add(Dense(self.input_size, activation="relu", input_dim=self.input_size))
        for v in self.hidden_size:
            model.add(Dense(v, activation="relu"))
        model.add(Dense(self.output_size, activation="softmax"))
        model.compile(optimizer=Adam(learning_rate=self._learning_rate), loss=MeanSquaredError(), run_eagerly=True)
        return model

    def action(self, state, legal_actions, game):
        actions = self.nn(np.array([state]))[0].numpy()
        actions = actions * game.legal_moves_binary()
        return np.unravel_index(np.argmax(actions), shape=game.state.shape)

    def train(self, x: list[int], y: list[int]):
        self.nn.fit(np.array(x), np.array(y), verbose=0)


if __name__ == "__main__":
    from hex import HexGame

    game = HexGame(size=5)
    actor = Actor(input_size=len(game.flat_state), hidden_size=(50, 50), output_size=25, learning_rate=0.03)
    print(actor.action(game.flat_state, game.legal_moves, game))