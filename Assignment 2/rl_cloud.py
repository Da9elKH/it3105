import ray
ray.init(num_cpus=32, num_gpus=2)

import wandb
import time
import numpy as np
import random
from collections import deque
from environments import Hex
from networks import ANN, CNN
from mcts import MCTS
from agents import MCTSAgent, ANNAgent, CNNAgent
from misc import LiteModel
from tensorflow.keras import Sequential
from config import App

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)

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


@ray.remote(num_cpus=1, num_gpus=0)
class Storage:
    def __init__(self):
        self.data = {"terminate": False, "checkpoint": -1, "num_played_steps": 0, "epsilon": App.config("mcts.epsilon")}

    def get_info(self, key):
        return self.data[key]

    def set_info(self, key, value):
        self.data[key] = value

    def all(self):
        return self.data


@ray.remote(num_cpus=1, num_gpus=0)
class Buffer:
    def __init__(self):
        self.buffer_train = deque([], maxlen=App.config("rbuf.queue_size"))
        self.buffer_test = []

        self.num_games = 0
        self.num_tot_samples = 0
        self.num_played_steps = 0
        self.num_tests = 0

    def store(self, game_history, storage):  # Game history
        if random.random() > 0.05 and self.num_tests != 0:
            self.buffer_train.append(game_history)
            self.num_games += 1
            self.num_tot_samples += len(self.buffer_train)
            self.num_played_steps += len(game_history.states)
            storage.set_info.remote("num_games", self.num_games)
        else:
            self.buffer_test.append(game_history)
            self.num_tests += 1

        self._solid_save(game_history)

    def get_batch(self, sample_size, test=False):
        if test:
            returns = np.random.choice(self.buffer_test, size=min(sample_size, len(self.buffer_test)), replace=False)
        else:
            returns = np.random.choice(self.buffer_train, size=min(sample_size, len(self.buffer_train)), replace=False)
        return returns

    def get_all(self, test=False):
        if test:
            returns = self.buffer_test
        else:
            returns = list(self.buffer_train)
        return returns

    def get_num_tot_samples(self):
        return self.num_tot_samples

    def get_num_games(self):
        return self.num_games

    def get_num_tests(self):
        return self.num_tests

    def _solid_save(self, game_history):
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


@ray.remote(num_gpus=1, num_cpus=2)
class Trainer:
    def __init__(self, network):
        self.network = network
        self.initialized = False
        self.training_step = 0

        wandb.login()
        wandb.init(project="hex", config={"mcts": App.config("mcts"), "cnn": App.config("cnn"), "rl": App.config("rl")})
        tf.debugging.set_log_device_placement(True)

    def loop(self, storage, buffer):
        if not self.initialized:
            self.initialize_ann(storage)

        while not ray.get(storage.get_info.remote("terminate")):
            num_tot_samples = ray.get(buffer.get_num_tot_samples.remote())
            num_games = ray.get(buffer.get_num_games.remote())
            num_tests = ray.get(buffer.get_num_tests.remote())
            start_time = time.time()

            if num_games > 1 and num_tests > 1:

                train = ray.get(buffer.get_batch.remote(App.config("rl.game_batch"), test=False))
                #test = ray.get(buffer.get_batch.remote(App.config("rl.game_batch"), test=True))
                #train = ray.get(buffer.get_all.remote(test=False))
                test = ray.get(buffer.get_all.remote(test=True))

                train_x, test_x = np.array(train[0].states), np.array(test[0].states)
                train_y, test_y = np.array(train[0].distributions), np.array(test[0].distributions)

                for i in range(1, len(train)):
                    train_x = np.append(train_x, train[i].states, 0)
                    train_y = np.append(train_y, train[i].distributions, 0)

                for i in range(1, len(test)):
                    test_x = np.append(test_x, test[i].states, 0)
                    test_y = np.append(test_y, test[i].distributions, 0)

                epochs = App.config("rl.epochs")

                #train_results = self.network.train_on_batch(train_x.astype(np.float32), train_y.astype(np.float32), None)
                train_results = self.network.fit(train_x.astype(np.float32), train_y.astype(np.float32), batch_size=len(train_x), epochs=epochs)
                test_results = self.network.model.evaluate(test_x.astype(np.float32), test_y.astype(np.float32), batch_size=len(test))

                epsilon = ray.get(storage.get_info.remote("epsilon"))

                wandb.log({
                    "accuracy": train_results.history["accuracy"][epochs-1],
                    "loss": train_results.history["loss"][epochs-1],
                    "kullback_leibler_divergence": train_results.history["kullback_leibler_divergence"][epochs-1],
                    "test_kullback_leibler_divergence": test_results[2],
                    "test_accuracy": test_results[1],
                    "test_loss": test_results[0],
                    "samples": num_tot_samples,
                    "games": num_games,
                    "tests": num_tests,
                    "training_step": self.training_step,
                    "epsilon": epsilon,
                    "training_time": time.time() - start_time,
                })

                weights = self.network.model.get_weights()
                storage.set_info.remote("nn_weights", weights)
                storage.set_info.remote("checkpoint", random.getrandbits(64))

                self.training_step += 1

                if self.training_step % 5 == 0:
                    storage.set_info.remote("epsilon", max(0.05, epsilon*0.99))
                    self.save(num_games)

                while ray.get(buffer.get_num_games.remote())/max(1, self.training_step) < App.config("rl.new_games_per_training_step"):
                    time.sleep(0.5)
            else:
                time.sleep(2)

    def save(self, num_games):
        model_name = f"S{App.config('environment.size')}_B{num_games}"
        self.network.save_model(model_name)

    def initialize_ann(self, storage):
        weights = self.network.model.get_weights()
        config = self.network.model.get_config()
        storage.set_info.remote("nn_weights", weights)
        storage.set_info.remote("nn_config", config)
        storage.set_info.remote("checkpoint", random.getrandbits(64))
        self.initialized = True
        self.save(0)


@ray.remote(num_cpus=1, num_gpus=0)
class MCTSWorker:
    def __init__(self, size, model):
        self.initialized = False
        self.environment = Hex(size=App.config("environment.size"))
        self.model = LiteModel.from_keras_model(model)

        if App.config("rl.use_cnn"):
            self.network = CNN(model=self.model)
            self.network_agent = CNNAgent(environment=self.environment, network=self.network)
        else:
            self.network = ANN(model=self.model)
            self.network_agent = ANNAgent(environment=self.environment, network=self.network)

        self.state_fc = self.network_agent.state_fc
        self.mcts = MCTS(
            environment=self.environment,
            rollout_policy_agent=self.network_agent,
            use_time_budget=App.config("mcts.use_time_budget"),
            rollouts=App.config("mcts.searches"),
            c=App.config("mcts.c"),
            epsilon=App.config("mcts.epsilon")
        )
        self.agent = MCTSAgent(environment=self.environment, mcts=self.mcts)
        self.checkpoint = None

    def updates_model_and_hyper_params(self, storage, checkpoint):
        if checkpoint != self.checkpoint:
            weights = ray.get(storage.get_info.remote("nn_weights"))
            config = ray.get(storage.get_info.remote("nn_config"))
            epsilon = ray.get(storage.get_info.remote("epsilon"))

            seq_model = Sequential.from_config(config)
            seq_model.set_weights(weights)

            self.model.update_keras_model(seq_model)
            self.checkpoint = checkpoint
            self.mcts.epsilon = epsilon

    def loop(self, storage, buffer):
        while not ray.get(storage.get_info.remote("terminate")):
            checkpoint = ray.get(storage.get_info.remote("checkpoint"))

            if checkpoint == -1:
                time.sleep(1)
            else:
                self.updates_model_and_hyper_params(storage, checkpoint)
                self.environment.reset()
                gh = GameHistory()
                while not self.environment.is_game_over:
                    move, distribution = self.agent.get_move(greedy=False)

                    # Normal state
                    gh.register_move(self.environment.current_player, move, self.state_fc(self.environment, rotate=False), distribution, self.environment.flat_state)

                    # Rotates state
                    size = self.environment.size
                    rotated_move = (size-1-move[0], size-1-move[1])
                    gh.register_move(self.environment.current_player, rotated_move, self.state_fc(self.environment, rotate=True), distribution[::-1], [self.environment.flat_state[0], *self.environment.flat_state[:0:-1]])

                    self.environment.play(move)
                buffer.store.remote(gh, storage)


if __name__ == "__main__":
    env = Hex(size=App.config("environment.size"))

    if App.config("rl.use_cnn"):
        network = CNN.build_from_config(input_shape=env.cnn_state.shape, output_size=len(env.legal_binary_moves))
    else:
        network = ANN.build(
            input_size=len(env.ann_state),
            output_size=len(env.legal_binary_moves),
            learning_rate=App.config("ann.learning_rate"),
            activation=App.config("ann.activation"),
            optimizer=App.config("ann.optimizer"),
            hidden_size=App.config("ann.hidden_layers")
        )

    buffer = Buffer.remote()
    storage = Storage.remote()
    trainer = Trainer.remote(network)
    workers = [MCTSWorker.remote(env.size, network.model) for _ in range(28)]

    # Run loops
    trainer.loop.remote(storage, buffer)
    for worker in workers:
        worker.loop.remote(storage, buffer)

    while True:
        time.sleep(10)
        print(f"Num samples: {ray.get(buffer.get_num_tot_samples.remote())}")
