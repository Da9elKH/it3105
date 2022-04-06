import logging
from functools import reduce


class App:
    _conf = {                                               # --- MAIN CONFIGS ---
        "run": [],                                          # Main tasks [ RL | TOPP | OHT ]
        "hex": {                                            # --- HEX BOARD GAME ---
            "log_level": logging.INFO,                      # Print log messages with this level
            "size": 7,                                      # Size of the Hex Board
        },
        "rl": {                                             # --- REINFORCEMENT LEARNER ---
            "visualize": False,                             # Render the board [ True | False]
            "use_cnn": True,                                # Train with CNN instead of ANN
            "epochs": 1,                                    # Number of epochs on training
            "game_batch": 3,                                # Number of games in each training batch
            "new_games_per_training_step": 7                # Regulates when training runs (Should be higher than game_batch to "see" all samples)
        },
        "topp": {                                           # --- TOURNAMENT OF PROGRESSIVE POLICIES ---
            "visualize": True,                              # Render the board [ True | False]
            "matches": 50,                                  # Matches between a pair of agents as each player
        },
        "oht": {                                            # --- ONLINE HEX TOURNAMENT ---
            "visualize": True,                              # Render the board [ True | False]
            "mode": "league",                               # Competition mode [ qualifiers | league ]
            "qualify": False,                               # Qualify towards subject grading
            "auth": "e1431af64ca24ffa9f2f3887e6b41a32",     # Authentication key
            "agent": "(1) CNN_S7_B1638.h5",                 # Agent to use in OHT
        },
        "cnn": {                                            # --- CONVOLUTIONAL NEURAL NETWORK ---
            "learning_rate": 0.001,                         # Alpha
            "hidden_layers": (64, 64, 64, 64, 64),          # Hidden conv-layers
            "activation": "relu",                           # Activation function [relu, ...]
            "optimizer": "adam",                            # Optimizer [adam, ...],
            "reg_const": 0.0001,                            # Kernel regularization "penalty"
            "temperature": 1                                # Temperature used on distribution
        },
        "ann": {                                            # --- ARTIFICIAL (DENSE) NEURAL NETWORK ---
            "learning_rate": 0.01,                          # Alpha
            "hidden_layers": (100, 100, 50),                # Hidden dense-layers
            "activation": "relu",                           # Activation function
            "optimizer": "adam",                            # Optimizer
            "temperature": 1                                # Temperature used on distribution
        },
        "mcts": {                                           # --- MONTE CARLO TREE SEARCH ---
            "log_level": logging.INFO,                      # Print log messages with this level
            "use_time_budget": False,                       # Use time budget instead of rollouts
            "time_budget": 0.1,                             # Time budget (in seconds)
            "searches": 5000,                               # MCTS-searches
            "c": 1.4,                                       # Exploration constant
            "epsilon": 1,                                   # Percentage of random moves in rollout
            "temperature": 0.25                             # Temperature used on distribution
        },
        "rbuf": {                                           # --- RBUF ---
            "load_samples": False,                          # Initialize samples
            "queue_size": 100,                              # Buffer-size (FiFo)
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
