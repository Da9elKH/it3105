import ray
import numpy as np
from agents import MCTSAgent
from memory import Memory
from environments import HexGame
from mcts import MCTS
from networks import CNN


@ray.remote
class MCTSWorker:
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


@ray.remote
class Trainer:
    def __init__(self):
        self.network = CNN.build(learning_rate=0.003)


if __name__ == "__main__" and False:
    ray.init(num_cpus=3)

    players = []
    actions = []
    states = []
    dists = []
    results = []

    # Initialize workers
    workers = [MCTSWorker.options(num_cpus=1).remote() for i in range(3)]

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

if __name__ == "__main__":

    import ray
    ray.init()

    import numpy as np
    import time
    import random

    @ray.remote
    class Storage:
        def __init__(self):
            self.checkpoint = {}
            self.calculations = {}

        def get_info(self, key):
            return self.checkpoint[key]

        def set_info(self, key, value):
            self.checkpoint[key] = value

        def all_calculations(self):
            return self.calculations

        def update_calculations(self, data):
            self.calculations.update(data)

        def add_calculation(self, data):
            key = random.getrandbits(64)
            self.calculations.update({key: {"finished": False, "data": data}})
            return key

        def get_calculation(self, key):
            if self._calculation_finished(key):
                return self.calculations[key]["data"]
            else:
                return None

        def _calculation_finished(self, key):
            if key in self.calculations:
                return self.calculations[key]["finished"]
            else:
                return False

        def all(self):
            return self.checkpoint

    @ray.remote
    class Predictor:
        def __init__(self):
            self.waiting_storage = {}
            self.finished_storage = {}
            self.BATCH_SIZE = 2

        def calculate(self, storage):
            waiting_storage = ray.get(storage.all_calculations.remote())

            if len(waiting_storage) > 0:
                # Unzip all values
                values = []
                keys = []
                for k, v in waiting_storage.items():
                    keys.append(k)
                    values.append(v["data"])

                # Calculation
                values = list((np.array(values) ** 2).flatten())

                # Store finished storage
                results = {}
                for i in range(len(values)):
                    results[keys[i]] = {"finished": True, "data": values[i]}
                storage.update_calculations.remote(results)

        def loop(self, storage):
            while True:
                time.sleep(0.05)
                self.calculate(storage)

        def all(self):
            return self.waiting_storage

    @ray.remote
    class MCTSWorker:
        def loop(self, storage):
            while True:
                number = random.randint(1, 100)
                key = storage.add_calculation.remote(number)
                while not (result := ray.get(storage.get_calculation.remote(key))):
                    time.sleep(0.05)
                storage.set_info.remote(number, result)


    storage = Storage.remote()
    predictor = Predictor.remote()
    mcts_workers = [MCTSWorker.remote() for _ in range(100)]

    [worker.loop.remote(storage) for worker in mcts_workers]
    predictor.loop.remote(storage)

    while True:
        print(len(ray.get(storage.all.remote())))
        time.sleep(0.5)
