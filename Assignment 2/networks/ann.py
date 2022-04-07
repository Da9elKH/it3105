from typing import Tuple
from tensorflow.keras import Sequential
from tensorflow.keras.activations import get as activation_get
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import get as optimizer_get
from config import App, ROOT_DIR
from .network import Network
import logging
logger = logging.getLogger(__name__)
logger.setLevel(App.config("ann.log_level"))
MODELS_FOLDER = f"{ROOT_DIR}/models/"


class ANN(Network):
    @classmethod
    def build_from_config(cls, input_size: int, output_size: int, **params):
        config = {"input_size": input_size, "output_size": output_size, **App.config("ann"), **params}
        return cls.build(**config)

    @classmethod
    def build(cls, input_size: int, hidden_layers: Tuple[int, ...], output_size: int, learning_rate: float, activation: str, optimizer: str, **params):
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
        for i, v in enumerate(hidden_layers):
            model.add(Dense(v, activation=act_fn, name=f"dense_{i+1}"))
        model.add(Dense(output_size, activation="softmax", name=f"dense_{len(hidden_layers) + 1}"))
        model.compile(optimizer=opt, loss=CategoricalCrossentropy(), metrics=["accuracy"])

        logger.info("Created following model:")
        model.summary(print_fn=logger.info)

        return cls(model=model)
