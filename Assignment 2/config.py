from pathlib import Path
from dotenv import load_dotenv
import logging
from functools import reduce
import os
import math
ROOT_DIR = Path(__file__).parent
load_dotenv(dotenv_path=Path(f"{ROOT_DIR}/secrets.env"))
CPUS = os.cpu_count()
INF = math.inf


class App:
    _conf = {                                               # === MAIN CONFIGS ===
        "run": ["RL", "TOPP"],                              # Main tasks [ RL | TOPP | OHT ]
        "cpus": CPUS//2,                                    # CPU count (includes "virtual" hyper-threaded cpus)
        "environment": {                                    # --- HEX BOARD GAME ---
            "log_level": logging.WARNING,                   # Print log messages with this level
            "size": 5,                                      # Size of the Hex Board
        },
        "rl": {                                             # === REINFORCEMENT LEARNER ===
            "log_level": logging.INFO,                      # Print log messages with this level
            "track": False,                                 # Wandb.ai tracking of process [ True | False ]
            "visualize": True,                              # Render the board [ True | False ]
            "use_cnn": True,                                # Train with CNN instead of ANN [ True | False ]
            "epochs": 1,                                    # Number of epochs on training
            "game_batch": 3,                                # Number of games in each training batch
            "training_steps": 15,                           # Number of training steps (episodes) to run (INF for inf)
            "epsilon_decay": 0.99,                          # Epsilon decay per training step
            "new_games_per_training_step": 1,               # Regulates when training runs (!)( >game_batch )
            "persist_model_per_training_step": 5,           # Persist model on each X 'th training step (INF for never)
        },
        "cnn": {                                            # === ARTIFICIAL (CONVOLUTIONAL) NEURAL NETWORK ===
            "log_level": logging.INFO,                      # Print log messages with this level
            "learning_rate": 0.003,                         # Alpha
            "hidden_layers": (64, 64, 64),                  # Hidden conv-layers (filters, )
            "activation": "relu",                           # Activation function [relu | linear | sigmoid | tanh]
            "optimizer": "adam",                            # Optimizer [adam | adagrad | SGD | RMSProp],
            "reg_const": 0.0001,                            # Kernel regularization "penalty"
            "temperature": 1                                # Temperature used on distribution
        },
        "ann": {                                            # === ARTIFICIAL (DENSE) NEURAL NETWORK ===
            "log_level": logging.INFO,                      # Print log messages with this level
            "learning_rate": 0.003,                         # Alpha
            "hidden_layers": (100, 100, 50),                # Hidden dense-layers (neurons, )
            "activation": "relu",                           # Activation function [relu | linear | sigmoid | tanh]
            "optimizer": "adam",                            # Optimizer [adam | adagrad | SGD | RMSProp]
            "temperature": 1                                # Temperature used on distribution
        },
        "mcts": {                                           # === MONTE CARLO TREE SEARCH ===
            "log_level": logging.WARNING,                   # Print log messages with this level
            "use_time_budget": False,                       # Use time budget instead of rollouts
            "time_budget": 1.0,                             # Time budget (in seconds)
            "searches": 1500,                               # MCTS-searches
            "c": 1.4,                                       # Exploration constant
            "epsilon": 1.0,                                 # Percentage of random moves in rollout
            "temperature": 0.25                             # Temperature used on distribution
        },
        "rbuf": {                                           # === REPLAY BUFFER ===
            "load_samples": False,                          # Initialize samples
            "save_samples": False,                          # Should the samples be saved to a file
            "queue_size": 100,                              # Buffer-size (FiFo)
            "test_size": 0.0                                # Percentage of all games being saved for evaluation
        },
        "topp": {                                           # === TOURNAMENT OF PROGRESSIVE POLICIES ===
            "log_level": logging.INFO,                      # Print log messages with this level
            "visualize": False,                             # Render the board [ True | False]
            "matches": 50,                                  # Matches between a pair of agents as each player (Double Round Robin)
            "models": [                                     # List of agents to play
                "5x5/(1) CNN_S5_B0.h5",
                "5x5/(1) CNN_S5_B75.h5",
                "5x5/(1) CNN_S5_B150.h5",
                "5x5/(1) CNN_S5_B225.h5",
                "5x5/(1) CNN_S5_B300.h5",
                #"5x5/(1) CNN_S5_B1950.h5"
            ],
        },
        "oht": {                                            # === ONLINE HEX TOURNAMENT ===
            "log_level": logging.INFO,                      # Print log messages with this level
            "qualify": False,                               # Qualify towards subject grading
            "visualize": False,                             # Render the board [ True | False]
            "mode": "qualifiers",                           # Competition mode [ qualifiers | league ]
            "auth":  os.getenv("OHT_AUTH"),                 # Authentication key (in separate secrets.env-file)
            "model": "7x7/(1) CNN_S7_B1638.h5",             # Agent to use in OHT
        },
    }

    @staticmethod
    def config(name):
        keys = App._keys(name)
        value = reduce(dict.get, keys, App._conf)

        if value is None:
            raise ValueError(f"Path '{name}' is not present in config")
        else:
            return value

    @staticmethod
    def set(name, value):
        keys = App._keys(name)
        App._nested_set(keys, value)

    @staticmethod
    def _nested_set(keys, value):
        dic = App._conf
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    @staticmethod
    def _keys(name):
        return name.split(".")
