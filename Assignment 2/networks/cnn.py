from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Input, add, ReLU, Softmax
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from os import path
import numpy as np

#import wandb
#wandb.init(project='hex')
#from wandb.keras import WandbCallback


MODELS_FOLDER = "/Users/daniel/Documents/AIProg/Assignments/Assignment 2/models/"


class CNN:
    def __init__(self, model: Model):
        self.model = model

    def predict(self, x):
        return self.model(x), None

    def train_on_batch(self, states, distributions, results):
        x = states
        y = distributions

        if not isinstance(states, np.ndarray):
            x = np.array(states)
        if not isinstance(distributions, np.ndarray):
            y = np.array(distributions)

        return self.model.train_on_batch(x, y, return_dict=True)

    def fit(self, x, y):
        self.model.fit(x, y, batch_size=256, epochs=50)#, callbacks=[WandbCallback()])


    @classmethod
    def from_file(cls, filename):
        """ Returns a network instance from file """
        model = load_model(MODELS_FOLDER + filename)
        return cls(model=model)

    @classmethod
    def build(cls, learning_rate: float, input_shape: tuple):
        """ Initialize the NN with the given depth and width for the problem environment """
        model = Sequential()
        model.add(Input(shape=(7, 7, 5)))

        # Convolutional layers
        for _ in range(4):
            model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', data_format="channels_last"))
            model.add(ReLU())

        # Policy layer
        model.add(Conv2D(filters=1, kernel_size=(1, 1), padding='same', data_format="channels_last"))
        model.add(Flatten())
        model.add(Dense(49))
        model.add(Softmax())

        model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])

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
