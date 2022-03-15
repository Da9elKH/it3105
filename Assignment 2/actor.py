from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from misc.state_manager import StateManager
import numpy as np
from os import path

MODELS_FOLDER = "models/"

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
        model.compile(optimizer=Adam(learning_rate=self._learning_rate), loss=CategoricalCrossentropy(), metrics=["accuracy"])
        return model

    def best_move(self, environment: StateManager):
        props = self.nn(np.array([environment.flat_state]))[0].numpy()
        props = props * environment.legal_binary_moves
        props = props / sum(props)
        return environment.transform_binary_move_index_to_move(binary_move_index=np.argmax(props))

    def train(self, x, y):
        self.nn.train_on_batch(np.array(x), np.array(y))

    def save_model(self, suffix):
        num = 1
        name = lambda n: "(%d) " % n + suffix + ".h5"

        if self.nn is not None:
            while path.exists(MODELS_FOLDER + name(num)):
                num += 1
            self.nn.save(MODELS_FOLDER + name(num))
        return name(num)

    def load_saved_model(self, name):
        self.nn = load_model(MODELS_FOLDER + name)
