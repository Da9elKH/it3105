import ray
import numpy as np
from agents import MCTSAgent
from memory import Memory

@ray.remote
class Worker:
    def __init__(self):

        # Cython implementation of board
        import pyximport
        pyximport.install(
            language_level=3,
            setup_args={
                "include_dirs": [np.get_include()],
                "define_macros": [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
            }
        )
        from rust_hex import modules

        self.memory = Memory(queue_size=1000, sample_size=1)
        self.environment = modules.HexGame(size=7)
        self.agent = MCTSAgent(
            environment=self.environment,
            model=modules.MCTS(
                environment=self.environment,
                time_budget=1.5,
                rollouts=0,
                c=1.4,
                verbose=True
            )
        )

    def sample_train(self):
        self.environment.reset()
        self.memory.reset()
        return self.run()

    def run(self):
        while not self.environment.is_game_over:
            # Run MCTS
            best_move, distribution = self.agent.get_move(greedy=True)

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

        print(players)

        filename = "/Users/daniel/Documents/AIProg/Assignments/Assignment 2/cases/train_samples"
        np.savetxt(filename + '_players.txt', players)
        np.savetxt(filename + '_actions.txt', actions)
        np.savetxt(filename + '_states.txt', states)
        np.savetxt(filename + '_dists.txt', dists)
        np.savetxt(filename + '_results.txt', results)

        print(f"Saved {len(players)} samples to file")
