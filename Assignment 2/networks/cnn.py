from typing import Tuple
from tensorflow.keras import Sequential, layers
from tensorflow.keras.activations import get as activation_get
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Input, Softmax
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import get as optimizer_get
from tensorflow.keras.regularizers import l2
from config import App, ROOT_DIR
from .network import Network
import logging
logger = logging.getLogger(__name__)
logger.setLevel(App.config("cnn.log_level"))
MODELS_FOLDER = f"{ROOT_DIR}/models/"


class CNN(Network):

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

        logger.info("Created following model:")
        model.summary(print_fn=logger.info)

        return cls(model=model)
