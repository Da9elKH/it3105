import ray
ray.init(num_cpus=32)

import wandb
import time
import numpy as np
import random
from environments import HexGame
from networks import ANN
from mcts import MCTS
from agents import MCTSAgent, ANNAgent
from misc import LiteModel
from tensorflow.keras import Sequential
from config import App


class GameHistory:
    def __init__(self):
        self.result = 0
        self.players = []
        self.states = []
        self.distributions = []
        self.moves = []

    def register_result(self, result):
        self.result = result

    def register_move(self, player, move, state, distribution):
        self.players.append(player)
        self.states.append(state)
        self.moves.append(move)
        self.distributions.append(distribution)

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
        self.data = {"terminate": False, "checkpoint": -1, "num_played_steps": 0, "epsilon": App.config("mcts.epsilon")}

    def get_info(self, key):
        return self.data[key]

    def set_info(self, key, value):
        self.data[key] = value

    def all(self):
        return self.data


@ray.remote
class Buffer:
    def __init__(self):
        self.buffer = {}
        self.test_buffer = {}

        self.num_games = 0
        self.num_samples = 0
        self.num_played_steps = 0

        self.num_tests = 0

    def store(self, game_history, storage):  # Game history
        if random.random() > 0.05:
            self.buffer[self.num_games] = game_history
            self.num_games += 1
            self.num_samples += len(game_history.states)
            self.num_played_steps += len(game_history.states)
            storage.set_info.remote("num_games", self.num_games)
        else:
            self.test_buffer[self.num_games] = game_history
            self.num_tests += 1

    def get_batch(self, sample_size, test=False):
        if test:
            keys = np.random.choice(list(self.test_buffer.keys()), size=min(sample_size, len(self.test_buffer)), replace=False)
            returns = [self.test_buffer[key] for key in keys]
        else:
            keys = np.random.choice(list(self.buffer.keys()), size=min(sample_size, len(self.buffer)), replace=False)
            returns = [self.buffer[key] for key in keys]
        return returns

    def get_num_samples(self):
        return self.num_samples

    def get_num_games(self):
        return self.num_games

    def get_num_tests(self):
        return self.num_tests


@ray.remote
class Trainer:
    def __init__(self, network):
        self.network = network
        self.initialized = False
        self.training_step = 0

        wandb.login()
        wandb.init(project="hex")

    def loop(self, storage, buffer):
        if not self.initialized:
            self.initialize_ann(storage)

        while not ray.get(storage.get_info.remote("terminate")):
            num_samples = ray.get(buffer.get_num_samples.remote())
            num_games = ray.get(buffer.get_num_games.remote())
            num_tests = ray.get(buffer.get_num_tests.remote())

            if num_games > 1 and num_tests > 1:

                train = ray.get(buffer.get_batch.remote(num_games//2, test=False))
                test = ray.get(buffer.get_batch.remote(num_tests//2, test=True))

                train_x, test_x = np.array(train[0].states), np.array(test[0].states)
                train_y, test_y = np.array(train[0].distributions), np.array(test[0].distributions)

                for i in range(1, len(train)):
                    train_x = np.append(train_x, train[i].states, 0)
                    train_y = np.append(train_y, train[i].distributions, 0)

                for i in range(1, len(test)):
                    test_x = np.append(test_x, test[i].states, 0)
                    test_y = np.append(test_y, test[i].distributions, 0)

                train_results = self.network.train_on_batch(train_x.astype(np.float32), train_y.astype(np.float32), None)
                test_results = self.network.model.evaluate(test_x.astype(np.float32), test_y.astype(np.float32), batch_size=len(test))

                epsilon = ray.get(storage.get_info.remote("epsilon"))

                wandb.log({
                    "accuracy": train_results["accuracy"],
                    "loss": train_results["loss"],
                    "test_accuracy": test_results[1],
                    "test_loss": test_results[0],
                    "samples": num_samples,
                    "games": num_games,
                    "tests": num_tests,
                    "training_step": self.training_step,
                    "epsilon": epsilon
                })

                weights = self.network.model.get_weights()
                storage.set_info.remote("nn_weights", weights)
                storage.set_info.remote("checkpoint", random.getrandbits(64))

                self.training_step += 1

                if self.training_step % 5 == 0:
                    storage.set_info.remote("epsilon", max(0.05, epsilon*0.99))
                    self.save(num_samples)

                while ray.get(buffer.get_num_games.remote())/max(1, self.training_step) < 1:
                    time.sleep(0.5)
            else:
                time.sleep(2)

    def save(self, num_samples):
        model_name = f"S{App.config('hex.size')}_B{num_samples}"
        self.network.save_model(model_name)

    def initialize_ann(self, storage):
        weights = self.network.model.get_weights()
        config = self.network.model.get_config()
        storage.set_info.remote("nn_weights", weights)
        storage.set_info.remote("nn_config", config)
        storage.set_info.remote("checkpoint", random.getrandbits(64))
        self.initialized = True
        self.save(0)

@ray.remote
class MCTSWorker:
    def __init__(self, size, model):
        self.initialized = False
        self.environment = HexGame(size=App.config("hex.size"))
        self.model = LiteModel.from_keras_model(model)
        self.network = ANN(model=self.model)
        self.ann_agent = ANNAgent(environment=self.environment, network=self.network)
        self.mcts = MCTS(
            environment=self.environment,
            rollout_policy_agent=self.ann_agent,
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
                    gh.register_move(self.environment.current_player, move, self.environment.ann_state, distribution)
                    self.environment.play(move)
                buffer.store.remote(gh, storage)


if __name__ == "__main__":
    env = HexGame(size=App.config("hex.size"))
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
        print(f"Num samples: {ray.get(buffer.get_num_samples.remote())}")
