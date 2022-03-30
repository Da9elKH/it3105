from agents import ANNAgent, MCTSAgent, CNNAgent, RandomAgent, BufferAgent
from mcts import MCTS
from environments import HexGame, HexGUI
from networks import ANN, CNN
from memory import Memory
from tqdm import trange
from topp import TOPP
import random

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

    def run(self, visualize=False):
        if visualize:
            gui = HexGUI(environment=self.environment)
            gui.run_visualization_loop(lambda: self.train())
        else:
            self.train()

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
    rl = ReinforcementLearning(games=200)
    rl.run()
