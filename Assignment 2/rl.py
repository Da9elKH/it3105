import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import ray
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for device in gpus:
    tf.config.experimental.set_memory_growth(device, True)

import wandb
import time
import numpy as np
import random
from collections import deque
from environments import Hex, HexGUI
from networks import ANN, CNN
from mcts import MCTS
from agents import MCTSAgent, ANNAgent, CNNAgent
from misc import LiteModel
from tensorflow.keras import Sequential
from config import App
from environment import Environment

import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(App.config("rl.log_level"))

"""
    === Reinforcement Learner ===
    
    These classes work in parallel to generate data with Monte-Carlo Tree Search
    It consists of:
        1. Storage
        2. Buffer
        3. Learner
        4. Workers

    The Workers gets weights from the storage, and runs MCTS and saves the states to the buffer.
    The Learner takes a sample from the buffer, trains and stores the weights in the storage.
    This continues until the storage-flag "terminate" occurs, and the process finishes.

"""

class GameHistory:
    def __init__(self):
        self.result = 0
        self.players = []
        self.states = []
        self.distributions = []
        self.moves = []
        self.flat_states = []

    @property
    def results(self):
        return [self.result]*len(self.states)

    def register_result(self, result):
        self.result = result

    def register_move(self, player, move, state, distribution, flat_state):
        self.players.append(player)
        self.states.append(state)
        self.moves.append(move)
        self.distributions.append(distribution)
        self.flat_states.append(flat_state)

    def stack(self):
        memory = []
        for i in range(len(self.states)):
            player = self.players[i]
            move = self.moves[i]
            state = self.states[i]
            distribution = self.distributions[i]
            result = self.result
            memory.append([player, move, state, distribution, result])
        return memory


@ray.remote
class Storage:
    def __init__(self):
        self.data = {
            "terminate": False,
            "checkpoint": -1,
            "epsilon": App.config("mcts.epsilon"),
            "models": []
        }

    def get_info(self, key):
        return self.data[key]

    def set_info(self, key, value):
        self.data[key] = value

    def append_info(self, key, value):
        if key in self.data:
            self.data[key].append(value)
        else:
            self.data[key] = [value]

    def all(self):
        return self.data


@ray.remote
class Buffer:
    def __init__(self):
        self.buffer = deque([], maxlen=App.config("rbuf.queue_size"))
        self.buffer_test = []
        self.num_games = 0
        self.num_tot_samples = 0
        self.num_tests = 0

    def store(self, game_history, storage):
        if random.random() <= App.config("rbuf.test_size") or (App.config("rbuf.test_size") > 0 and self.num_tests == 0):
            self.buffer_test.append(game_history)
            self.num_tests += 1
        else:
            self.buffer.append(game_history)
            self.num_games += 1
            self.num_tot_samples += len(game_history.flat_states)
            storage.set_info.remote("num_games", self.num_games)
        self._solid_save(game_history)

    def get_batch(self, sample_size, test=False):
        if test:
            return np.random.choice(self.buffer_test, size=min(sample_size, len(self.buffer_test)), replace=False)
        else:
            return np.random.choice(self.buffer, size=min(sample_size, len(self.buffer)), replace=False)

    def get_all(self, test=False):
        if test:
            return self.buffer_test
        else:
            return list(self.buffer)

    def get_num_tot_samples(self):
        return self.num_tot_samples

    def get_num_games(self):
        return self.num_games

    def get_num_tests(self):
        return self.num_tests

    @staticmethod
    def _solid_save(game_history):
        if App.config("rbuf.save_samples"):
            with open("cases/flat_states.txt", "ab") as f:
                np.savetxt(f, game_history.flat_states)
            with open("cases/distributions.txt", "ab") as f:
                np.savetxt(f, game_history.distributions)
            with open("cases/results.txt", "ab") as f:
                np.savetxt(f, game_history.results)
            with open("cases/moves.txt", "ab") as f:
                np.savetxt(f, game_history.moves)
            with open("cases/players.txt", "ab") as f:
                np.savetxt(f, game_history.players)


@ray.remote(num_cpus=1, num_gpus=len(gpus))
class Trainer:
    def __init__(self, network):
        tf.debugging.set_log_device_placement(True)

        self.network = network
        self.initialized = False
        self.training_step = 0
        self.new_games_per_training = App.config("rl.new_games_per_training_step")
        self.epochs = App.config("rl.epochs")
        self.batch_size = App.config("rl.game_batch")
        self.epsilon_decay = App.config("rl.epsilon_decay")

        if App.config("rl.track"):
            wandb.login()
            wandb.init(project="hex", config={
                "mcts": App.config("mcts"),
                "cnn": App.config("cnn"),
                "rl": App.config("rl")
            })

    def loop(self, storage, buffer):
        # Metadata
        max_training_steps = np.inf if App.config("rl.training_steps") is None else App.config("rl.training_steps")

        # Initialize network weights in storage and set checkpoint
        if not self.initialized:
            self.initialize_ann(storage)

        # Wait for the first data to be stored in storage
        while ray.get(buffer.get_num_games.remote()) == 0 or (ray.get(buffer.get_num_tests.remote()) == 0 and App.config("rbuf.test_size") > 0):
            time.sleep(2)

        # While not terminated, run the training loop
        while not ray.get(storage.get_info.remote("terminate")):
            self.single_run(storage, buffer)

            # Terminate process if training_steps is completed
            if self.training_step >= max_training_steps:
                storage.set_info.remote("terminate", True)

            while (ray.get(buffer.get_num_games.remote()) / max(1, self.training_step)) <= self.new_games_per_training:
                time.sleep(0.5)

    def single_run(self, storage, buffer):
        # Metadata
        epsilon = ray.get(storage.get_info.remote("epsilon"))
        num_tot_samples = ray.get(buffer.get_num_tot_samples.remote())
        num_games = ray.get(buffer.get_num_games.remote())
        start_time = time.time()

        # Load training and test data and create numpy arrays
        train = ray.get(buffer.get_batch.remote(self.batch_size, test=False))
        train_x, train_y = np.array(train[0].states), np.array(train[0].distributions)

        for i in range(1, len(train)):
            train_x = np.append(train_x, train[i].states, 0)
            train_y = np.append(train_y, train[i].distributions, 0)

        # Run training of network
        self.training_step += 1
        train_results = self.network.fit(
            train_x.astype(np.float32),
            train_y.astype(np.float32),
            batch_size=len(train_x),
            epochs=self.epochs,
        )

        test_stats = {}
        if App.config("rbuf.test_size") > 0:
            num_tests = ray.get(buffer.get_num_tests.remote())
            test = ray.get(buffer.get_batch.remote(self.batch_size, test=True))
            test_x, test_y = np.array(train[0].states), np.array(train[0].distributions)

            # Run evaluation to track the performance
            test_results = self.network.model.evaluate(
                test_x.astype(np.float32),
                test_y.astype(np.float32),
                batch_size=len(test)
            )
            test_stats = {
                "test_loss": test_results[0],
                "test_accuracy": test_results[1],
                "test_kullback_leibler_divergence": test_results[2],
                "tests": num_tests,
            }

        # Track results
        if App.config("rl.track"):
            wandb.log({
                "accuracy": train_results.history["accuracy"][self.epochs - 1],
                "loss": train_results.history["loss"][self.epochs - 1],
                "kullback_leibler_divergence": train_results.history["kullback_leibler_divergence"][self.epochs - 1],
                "samples": num_tot_samples,
                "games": num_games,
                "training_step": self.training_step,
                "epsilon": epsilon,
                "training_time": time.time() - start_time,
                **test_stats
            })

        # Store weights, checkpoint and new epsilon
        storage.set_info.remote("nn_weights", self.network.model.get_weights())
        storage.set_info.remote("checkpoint", random.getrandbits(64))
        storage.set_info.remote("epsilon", max(0.05, epsilon * App.config("rl.epsilon_decay")))

        # Persist model
        if self.training_step % App.config("rl.persist_model_per_training_step") == 0:
            model_name = self.save(self.training_step)
            storage.append_info.remote("models", model_name)

    def save(self, episodes):
        model_name = f"S{App.config('environment.size')}_B{episodes}"
        return self.network.save_model(model_name)

    def initialize_ann(self, storage):
        weights = self.network.model.get_weights()
        config = self.network.model.get_config()
        storage.set_info.remote("nn_weights", weights)
        storage.set_info.remote("nn_config", config)
        storage.set_info.remote("checkpoint", random.getrandbits(64))
        self.initialized = True
        model_name = self.save(0)
        storage.append_info.remote("models", model_name)


@ray.remote(num_cpus=1, num_gpus=0)
class MCTSWorker:
    def __init__(self, model, size):
        self.initialized = False
        self.environment = Hex(size=size)
        self.model = LiteModel.from_keras_model(model)

        if App.config("rl.use_cnn"):
            self.network = CNN(model=self.model)
            self.network_agent = CNNAgent(environment=self.environment, network=self.network)
        else:
            self.network = ANN(model=self.model)
            self.network_agent = ANNAgent(environment=self.environment, network=self.network)

        self.checkpoint = None
        self.state_fc = self.network_agent.state_fc
        self.agent = MCTSAgent(
            environment=self.environment,
            mcts=MCTS.from_config(
                environment=self.environment,
                rollout_policy_agent=self.network_agent
            )
        )

    def loop(self, storage, buffer):
        # Wait for the Trainer to initialize the weights
        while ray.get(storage.get_info.remote("checkpoint")) == -1:
            time.sleep(1)

        # Render MCTS Workers
        if App.config("rl.visualize"):
            gui = HexGUI(height=400, width=400, environment=self.environment)
            gui.run_visualization_loop(lambda: self._loop(storage, buffer))
        else:
            self._loop(storage, buffer)

    def _loop(self, storage, buffer):
        # Main loop, run until terminated
        while not ray.get(storage.get_info.remote("terminate")):
            checkpoint = ray.get(storage.get_info.remote("checkpoint"))
            self.updates_model_and_hyper_params(storage, checkpoint)
            self.single_run(storage, buffer)

    def single_run(self, storage, buffer):
        self.environment.reset()
        game_history = GameHistory()

        while not self.environment.is_game_over:
            move, distribution = self.agent.get_move(greedy=False)

            # Metadata
            size = self.environment.size
            rotated_move = (size - 1 - move[0], size - 1 - move[1])
            flat_state = self.environment.flat_state
            rotated_flat_state = [flat_state[0], *flat_state[:0:-1]]

            # Save normal state
            game_history.register_move(
                self.environment.current_player,
                move,
                self.state_fc(self.environment, rotate=False),
                distribution,
                flat_state
            )

            # Save rotated state
            game_history.register_move(
                self.environment.current_player,
                rotated_move,
                self.state_fc(self.environment, rotate=True),
                distribution[::-1],
                rotated_flat_state
            )

            # Execute the move in the environment
            self.environment.play(move)

        # Store the game history in the buffer
        buffer.store.remote(game_history, storage)

    def updates_model_and_hyper_params(self, storage, checkpoint):
        if checkpoint != self.checkpoint:
            weights = ray.get(storage.get_info.remote("nn_weights"))
            config = ray.get(storage.get_info.remote("nn_config"))
            epsilon = ray.get(storage.get_info.remote("epsilon"))

            seq_model = Sequential.from_config(config)
            seq_model.set_weights(weights)

            self.model.update_keras_model(seq_model)
            self.checkpoint = checkpoint
            self.agent.mcts.epsilon = epsilon


class ReinforcementLearner:
    def __init__(self, environment: Environment):
        ray.init(num_cpus=App.config("cpus"))
        self.environment = environment.copy()
        self.saved_models = []

        if App.config("rl.use_cnn"):
            self.network = CNN.build_from_config(
                input_shape=self.environment.cnn_state.shape,
                output_size=len(self.environment.legal_binary_moves))
        else:
            self.network = ANN.build_from_config(
                input_size=len(self.environment.ann_state),
                output_size=len(self.environment.legal_binary_moves)
            )

        ts = App.config("rl.training_steps")
        self.training_steps = np.inf if ts is None else ts
        self.buffer = Buffer.remote()
        self.storage = Storage.remote()
        self.trainer = Trainer.remote(self.network)

        # With -2 and cpus = os.cpu_count, buffer and storage share cpu
        cpus = App.config("cpus") - 2
        self.workers = [MCTSWorker.remote(self.network.model, self.environment.size) for _ in range(cpus)]
        self.gui = None

    def run(self):
        # Run trainer loop
        self.trainer.loop.remote(self.storage, self.buffer)

        # Run worker loop
        for worker in self.workers:
            worker.loop.remote(self.storage, self.buffer)

        while not ray.get(self.storage.get_info.remote("terminate")):
            time.sleep(10)
            samples = ray.get(self.buffer.get_num_tot_samples.remote())
            logger.info(f"Current sample size: {samples}")

        self.saved_models = ray.get(self.storage.get_info.remote("models"))
        ray.shutdown()
        logger.info("RL Completed")
