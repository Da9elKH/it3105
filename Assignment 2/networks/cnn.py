from os import path
from typing import Tuple

import numpy as np
from tensorflow.keras import Sequential, layers
from tensorflow.keras.activations import get as activation_get
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Input, Softmax
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import get as optimizer_get
from tensorflow.keras.regularizers import l2

from config import App, ROOT_DIR
from misc import LiteModel

MODELS_FOLDER = f"{ROOT_DIR}/models/"


class CNN:
    def __init__(self, model: Model):
        self.model = model

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

    def fit(self, x, y, **params):
        return self.model.fit(x, y, **params)


    @classmethod
    def from_file(cls, filename):
        """ Returns a network instance from file """
        model = load_model(MODELS_FOLDER + filename)
        return cls(model=model)

    @classmethod
    def build_from_config(cls, input_shape: tuple, output_size: int, **params):
        config = {"input_shape": input_shape, "output_size": output_size, **App.config("cnn"), **params}
        return cls.build(**config)

    @classmethod
    def build(cls, learning_rate: float, input_shape: tuple, output_size: int, hidden_layers: Tuple[int] = (32, 32), activation: str = "relu", optimizer: str = "adam", reg_const: float = 0.0001, **args):
        """ Initialize the NN with the given depth and width for the problem environment """

        act_fn = activation_get(activation)
        opt = optimizer_get(
            {
                "class_name": optimizer,
                "config": {
                    "lr": learning_rate,
                }
            }
        )

        model = Sequential()
        model.add(Input(shape=input_shape))

        # Convolutional layers
        for i in range(0, len(hidden_layers)):
            kernel_size = (5, 5) if i == 0 else (3, 3)
            model.add(Conv2D(
                filters=hidden_layers[i],
                kernel_size=kernel_size,
                kernel_regularizer=l2(reg_const),
                padding='same',
                data_format="channels_last",
                name=f"conv_{i}")
            )
            model.add(BatchNormalization(axis=1, name=f"batch_{i}"))
            model.add(layers.Activation(act_fn, name=f"act_{i}"))

        # Policy layer
        model.add(Conv2D(
            filters=1,
            kernel_size=(1, 1),
            kernel_regularizer=l2(reg_const),
            padding='same',
            data_format="channels_last",
            name=f"conv_{len(hidden_layers)}")
        )
        model.add(Flatten(name=f"flat_{len(hidden_layers)}"))
        model.add(Softmax())
        model.compile(loss=CategoricalCrossentropy(), optimizer=opt, metrics=["accuracy", "KLDivergence"])

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
