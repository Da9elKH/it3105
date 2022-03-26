from agents import ANNAgent, MCTSAgent, CNNAgent
from mcts import MCTS
from environments import HexGame
from networks import ANN, CNN
from memory import Memory
from tqdm import trange
from topp import TOPP


class ReinforcementLearning:
    def __init__(self, games):
        self.games = games
        self.models = []

        self.environment = HexGame(size=7)
        self.network = ANN.build(
            input_size=len(self.environment.flat_state),
            output_size=len(self.environment.legal_binary_moves),
            hidden_size=(200, 100, 50),
            learning_rate=0.03
        )
        """
        self.network = CNN.build(
            input_size=self.environment.cnn_state.shape,
            output_size=len(self.environment.legal_binary_moves),
            hidden_size=(200, 100, 50),
            learning_rate=0.1,
            momentum=0.9
        )
        """

        self.mcts = MCTS(
            rollout_policy_agent=ANNAgent(network=self.network),
            environment=self.environment,
            rollouts=1500,
            time_budget=5,
            epsilon=1.00,
            verbose=False,
            c=1.5
        )
        self.agent = MCTSAgent(
            environment=self.environment,
            model=self.mcts,
        )
        self.memory = Memory(
            sample_size=0.25,
            queue_size=256,
            verbose=False
        )

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
                    self.memory.register_state_and_distribution(self.environment.flat_state, distribution)

                    # Play the move
                    self.environment.play(best_move)

                # Register result of game in memory
                self.memory.register_result(self.environment.result)

                # Train actor on memory
                self.network.train_on_batch(*self.memory.sample())

                if i % 25 == 0:
                    self.save_model(i)

                # Epsilon decay
                self.mcts.epsilon *= 0.9955

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
    rl = ReinforcementLearning(games=1000)
    rl.train()
