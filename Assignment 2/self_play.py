import ray
import numpy as np
from agents import MCTSAgent
from memory import Memory
from environments import HexGame
from mcts import MCTS


@ray.remote
class Worker:
    def __init__(self):

        self.memory = Memory(queue_size=1000, sample_size=1)
        self.environment = HexGame(size=7)
        self.agent = MCTSAgent(
            environment=self.environment,
            mcts=MCTS(
                environment=self.environment,
                use_time_budget=False,
                rollouts=1000,
                c=1.4,
            )
        )

    def sample_train(self):
        self.environment.reset()
        self.memory.reset()
        return self.run()

    def run(self):
        while not self.environment.is_game_over:
            # Run MCTS
            best_move, distribution = self.agent.get_move(greedy=False)

            # Add state and distribution to memory
            self.memory.register("player", self.environment.current_player)
            self.memory.register("action", best_move)
            self.memory.register("state", self.environment.flat_state)
            self.memory.register("distribution", distribution.flatten().tolist())

            # Play the move
            self.environment.play(best_move)

        # Register result of game in memory
        self.memory.register_result(self.environment.result)

        return self.memory.all()


if __name__ == "__main__":
    ray.init(num_cpus=3)

    players = []
    actions = []
    states = []
    dists = []
    results = []

    # Initialize workers
    workers = [Worker.options(num_cpus=1).remote() for i in range(3)]

    while True:
        # Run workers async
        tasks = [worker.sample_train.remote() for worker in workers]

        # Await samples
        samples = ray.get(tasks)

        for sample in samples:
            players.extend(sample[0])
            actions.extend(sample[1])
            states.extend(sample[2])
            dists.extend(sample[3])
            results.extend(sample[4])

        filename = "/Users/daniel/Documents/AIProg/Assignments/Assignment 2/cases/train_samples"
        np.savetxt(filename + '_players.txt', players)
        np.savetxt(filename + '_actions.txt', actions)
        np.savetxt(filename + '_states.txt', states)
        np.savetxt(filename + '_dists.txt', dists)
        np.savetxt(filename + '_results.txt', results)

        print(f"Saved {len(players)} samples to file")
