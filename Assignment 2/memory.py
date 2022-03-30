from collections import deque
import numpy as np
import math
import random


class Memory:
    def __init__(self, queue_size=1028, sample_size=0.1, verbose=False):
        self._sample_size = sample_size
        self._lt_memory = deque([], maxlen=queue_size)
        self._st_memory = {
            "result": None,
            "state": [],
            "distribution": [],
            "player": [],
            "action": []
        }
        self.verbose = False

    def register(self, key: str, value):
        self._st_memory[key].append(value)

    def register_result(self, result):
        self._st_memory["result"] = result
        self._store()

    def register_state_and_distribution(self, state, distribution):
        self._st_memory["state"].append(state)
        self._st_memory["distribution"].append(distribution)

    def _store(self):
        length = len(self._st_memory["state"])

        if self.verbose:
            print(f"MEMORY: Storing {length} states to long-term-memory")

        # Store short term memory in long term memory
        for i in range(length):
            result = self._st_memory["result"]
            state = self._st_memory["state"][i]
            dist = self._st_memory["distribution"][i]
            player = self._st_memory["player"][i]
            action = self._st_memory["action"][i]

            self._lt_memory.append(
                [player, action, state, dist, result]
            )

        # Reset short term memory
        self._st_memory = {
            "result": None,
            "state": [],
            "distribution": [],
            "player": [],
            "action": []
        }

    def sample(self):
        samples = random.sample(self._lt_memory, k=math.ceil(len(self._lt_memory) * self._sample_size))
        players, actions, states, dists, results = zip(*samples)

        if self.verbose:
            print(f"MEMORY: Returning {len(states)} samples")

        return players, actions, states, dists, results

    def all(self):
        players, actions, states, dists, results = zip(*self._lt_memory)
        return list(players), list(actions), list(states), list(dists), list(results)

    def reset(self):
        self._lt_memory.clear()
