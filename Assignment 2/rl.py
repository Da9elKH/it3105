from agents import ANNAgent, MCTSAgent, CNNAgent
from environments import HexGame, HexGUI
from networks import ANN, CNN
from memory import Memory
from tqdm import trange
from topp import TOPP
from mcts import MCTS
from config import App
import random
import numpy as np
import matplotlib.pyplot as plt


class ReinforcementLearning:
    def __init__(self, games):
        self.games = games
        self.eps_decay = 0.05 ** (1. / self.games)

        self.models = []
        self.environment = HexGame(size=7)

        """
        self.network = ANN.build(
            optimizer="adam",
            activation="relu",
            learning_rate=0.03,
            input_size=len(self.environment.flat_state),
            output_size=len(self.environment.legal_binary_moves),
            hidden_size=(100, 100, 50),
        )
        """

        self.network = CNN.build(
            input_shape=self.environment.cnn_state.shape,
            learning_rate=0.001,
        )
        self.mcts = MCTS(
            rollout_policy_agent=CNNAgent(network=self.network),
            environment=self.environment,
            rollouts=1000,
            time_budget=1,
            epsilon=1,
            verbose=True,
            c=1.4
        )
        self.agent = MCTSAgent(
            environment=self.environment,
            model=self.mcts,
        )
        self.memory = Memory(
            sample_size=0.5,
            queue_size=10000,
            verbose=False
        )

    def run(self):
        if App.config("rl.visualize"):
            gui = HexGUI(environment=self.environment)
            gui.run_visualization_loop(lambda: self.train())
        else:
            self.train()


    def pre_train(self):
        pass


    def train(self):
        # Save model before training

        self.save_model(0)

        with trange(1, self.games + 1) as t:
            for i in t:

                t.set_description(f"Game {i}")
                self.environment.reset()

                while not self.environment.is_game_over:
                    # Run MCTS
                    best_move, distribution = self.agent.get_move(greedy=True)

                    # Add state and distribution to memory
                    self.memory.register_state_and_distribution(self.environment.cnn_state, distribution)

                    # Add rotated state and distribution to memory
                    if random.random() > 0.5:
                        self.memory.register_state_and_distribution(self.environment.rotated_cnn_state, distribution[::-1])

                    # Play the move
                    self.environment.play(best_move)

                # Register result of game in memory
                self.memory.register_result(self.environment.result)

                # Train actor on memory
                print(self.network.train_on_batch(*self.memory.sample()))

                if i % 5 == 0:
                    self.save_model(i)

                # Epsilon decay
                self.mcts.epsilon *= self.eps_decay

        self.check_models()

    def save_model(self, i):
        model_name = lambda size, batch: f"S{size}_B{i}"
        self.models.append(
            self.network.save_model(suffix=model_name(self.environment.size, self.games))
        )

    def check_models(self):
        environment = self.environment.copy()
        environment.reset()

        topp = TOPP(environment=environment)

        for filename in self.models:
            topp.add_agent(
                filename,
                ANNAgent(environment=environment, network=ANN.from_file(filename))
            )

        print(topp.tournament(100))


if __name__ == "__main__":
    #rl = ReinforcementLearning(games=200)
    #rl.run()

    def ann_preprocessing():
        filename = "/Users/daniel/Documents/AIProg/Assignments/Assignment 2/cases/r_5000_new2/train_samples"
        states = np.loadtxt(filename + '_states.txt', dtype=np.int32)
        dists = np.loadtxt(filename + '_dists.txt', dtype=np.float32)

        dict = {0: 0, 1: 1, -1: 2}
        bits = lambda s: format(dict[s], f"0{2}b")
        new_states = np.zeros((states.shape[0], 100))

        for i in range(len(states)):
            new_states[i] = np.array([float(s) for s in list(''.join([bits(s) for s in states[i]]))])

        return new_states.astype(np.int32), dists

    def preprocessing():
        filename = "/Users/daniel/Documents/AIProg/Assignments/Assignment 2/cases/r_5000_new2/train_samples"
        states = np.loadtxt(filename + '_states.txt', dtype=np.float32)
        dists = np.loadtxt(filename + '_dists.txt', dtype=np.float32)
        transpose_players = (states[:, 0] == -1)

        # State preprocessing
        flat_states = states[:, 1:].reshape((states.shape[0], 7, 7))
        flat_states[transpose_players] = np.transpose(flat_states[transpose_players], axes=(0, 2, 1)) * -1
        flat_states = np.array([flat_states == 1, flat_states == -1, flat_states == 0], dtype=np.float32)
        flat_states = np.moveaxis(flat_states, 0, 3)

        # Dists preprocessing
        dists = dists.reshape((dists.shape[0], 7, 7))
        dists[transpose_players] = np.transpose(dists[transpose_players], axes=(0, 2, 1))
        dists = dists.reshape((dists.shape[0], 49))

        return flat_states, dists

    def preprocessing_new():
        filename = "/Users/daniel/Documents/AIProg/Assignments/Assignment 2/cases/r_5000_new2/train_samples"
        states = np.loadtxt(filename + '_states.txt')
        dists = np.loadtxt(filename + '_dists.txt')

        player1 = (states[:, 0] == 1)
        player2 = (states[:, 0] == -1)

        # State preprocessing
        flat_states = states[:, 1:].reshape((states.shape[0], 7, 7))
        #flat_states[transpose_players] = np.transpose(flat_states[transpose_players], axes=(0, 2, 1)) * -1
        flat_states = np.array([flat_states == 1, flat_states == -1, flat_states == 0, np.zeros((states.shape[0], 7, 7)), np.zeros((states.shape[0], 7, 7))], dtype=np.float32)
        flat_states = np.moveaxis(flat_states, 0, 3)
        flat_states[player1, :, :, 3] = 1
        flat_states[player2, :, :, 4] = 1

        # Dists preprocessing
        #dists = dists.reshape((dists.shape[0], 7, 7))
        #dists[transpose_players] = np.transpose(dists[transpose_players], axes=(0, 2, 1))
        #dists = dists.reshape((dists.shape[0], 49))

        return flat_states, dists


    transpose = True
    states, dists = preprocessing()
    #states, dists = preprocessing_new()
    #states, dists = ann_preprocessing()

    env = HexGame(size=7)
    """
    ann = ANN.build(
        learning_rate=0.001,
        input_size=len(states[0]),
        hidden_size=(200, 100, 50),
        output_size=len(env.legal_binary_moves),
        activation="relu",
        optimizer="adam"

    )
    """
    cnn = CNN.build_from_config(input_shape=env.cnn_state.shape, output_size=len(env.legal_binary_moves))

    # Shuffle the data
    idx = np.random.permutation(len(states))
    states = states[idx]
    dist = dists[idx]

    # Train, test split
    #X_train, X_test, y_train, y_tes = train_test_split(states, dists, test_size=0.1, shuffle=True)

    #train_dataset = tf.data.Dataset.from_tensor_slices((states, dists)).shuffle(80000).batch(256)
    #test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_tes)).batch(256)

    # Plotting
    train_accuracies = []
    train_loss = []

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Accuracy")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Loss")

    i = 0
    for i in range(500):
        # Generate sample
        idx = np.arange(states.shape[0])
        batch_idx = np.random.choice(idx, 500, replace=False)

        result = cnn.fit(states[batch_idx], dists[batch_idx], epochs=10, batch_size=500)
        #result = cnn.train_on_batch(states[batch_idx], dists[batch_idx], None)

        loss = result.history["loss"][9]
        acc = result.history["accuracy"][9]

        train_accuracies.append(acc)
        train_loss.append(loss)
        print(f"Epoch {i}, loss: {loss}, acc: {acc}")

        i += 1

        if i % 20 == 0:
            x = np.arange(len(train_accuracies))
            ax1.plot(x, train_accuracies, color='tab:green', label="Train")
            ax2.plot(x, train_loss, color='tab:orange', label="Train")
            plt.show(block=False)
            plt.pause(0.001)

    plt.show()
    #cnn.save_model(f"sample_r5000_l{env.cnn_state.shape[2]}_T{transpose}")