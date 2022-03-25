from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from os import path
import numpy as np

MODELS_FOLDER = "/Users/daniel/Documents/AIProg/Assignments/Assignment 2/models/"

class ANN:
    def __init__(self, model: Model):
        self.model = model

    def predict(self, x):
        return self.model(x)

    def train_on_batch(self, states, distributions, results):
        x, y = np.array(states), np.array(distributions)
        return self.model.train_on_batch(x, y)

    @classmethod
    def from_file(cls, filename):
        """ Returns a network instance from file """
        model = load_model(MODELS_FOLDER + filename)
        return cls(model=model)

    @classmethod
    def build(cls, input_size: int, output_size: int, hidden_size: tuple[int, ...], learning_rate: float):
        """ Initialize the NN with the given depth and width for the problem environment """
        model = Sequential()
        model.add(Dense(input_size, activation="relu", input_dim=input_size))
        for v in hidden_size:
            model.add(Dense(v, activation="relu"))
        model.add(Dense(output_size, activation="softmax"))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=CategoricalCrossentropy(),
                      metrics=["accuracy"])
        return cls(model=model)

    """ MISC """
    def save_model(self, suffix):
        num = 1
        name = lambda n: "(%d) " % n + f"{self.__class__.__name__}_" + suffix + ".h5"

        if self.model is not None:
            while path.exists(MODELS_FOLDER + name(num)):
                num += 1
            self.model.save(MODELS_FOLDER + name(num))
        return name(num)

