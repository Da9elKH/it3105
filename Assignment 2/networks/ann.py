from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.activations import get as activation_get
from tensorflow.keras.optimizers import get as optimizer_get
from typing import Tuple
from os import path
from misc import LiteModel
from config import App
import numpy as np

MODELS_FOLDER = "models/"


class ANN:
    def __init__(self, model: Model, config=None):
        self.model = model
        self.config = {} if not config else config

    def predict(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if isinstance(self.model, LiteModel):
            return self.model.predict_single(x)
        else:
            return self.model(x)

    def train_on_batch(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        return self.model.train_on_batch(x, y, return_dict=True)

    def fit(self, states, distributions):
        x = states
        y = distributions

        if not isinstance(states, np.ndarray):
            x = np.array(states)
        if not isinstance(distributions, np.ndarray):
            y = np.array(distributions)

        return self.model.fit(x, y)

    @classmethod
    def from_file(cls, filename):
        """ Returns a network instance from file """
        model = load_model(MODELS_FOLDER + filename)
        return cls(model=model)

    @classmethod
    def build_from_config(cls, input_size: int, output_size: int, **params):
        config = {"input_size": input_size, "output_size": output_size, **App.config("ann"), **params}
        return cls(**config)

    @classmethod
    def build(cls, input_size: int, hidden_size: Tuple[int, ...], output_size: int, learning_rate: float, activation: str, optimizer: str):
        """ Initialize the NN with the given depth and width for the problem environment """

        # Dynamically get activation function and optimizer
        act_fn = activation_get(activation)
        opt = optimizer_get(
            {
                "class_name": optimizer,
                "config": {
                    "lr": learning_rate,
                }
            }
        )

        # Build model
        model = Sequential()
        model.add(Dense(input_size, activation=act_fn, input_dim=input_size, name="dense_0"))
        for i, v in enumerate(hidden_size):
            model.add(Dense(v, activation=act_fn, name=f"dense_{i+1}"))
        model.add(Dense(output_size, activation="softmax", name=f"dense_{len(hidden_size) + 1}"))
        model.compile(optimizer=opt, loss=CategoricalCrossentropy(), metrics=["accuracy"])

        config = {"lr": learning_rate, "activation": activation, "optimizer": optimizer, "i_size": input_size, "o_size": output_size, "h_layers": hidden_size}
        return cls(model=model, config=config)

    """ MISC """
    def save_model(self, suffix):
        num = 1
        name = lambda n: "(%d) " % n + f"{self.__class__.__name__}_" + suffix + ".h5"

        if self.model is not None:
            while path.exists(MODELS_FOLDER + name(num)):
                num += 1
            self.model.save(MODELS_FOLDER + name(num))
        return name(num)
