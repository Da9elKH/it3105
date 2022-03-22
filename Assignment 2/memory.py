from collections import deque
import numpy as np
import math
import random


class Memory:
    def __init__(self, queue_size=1028, sample_size=0.1):
        self._sample_size = sample_size
        self._lt_memory = deque([], maxlen=queue_size)
        self._st_memory = {
            "result": None,
            "states": [],
            "distributions": [],
        }

    def register_result(self, result):
        self._st_memory["result"] = result
        self._store()

    def register_state_and_distribution(self, state, distribution):
        self._st_memory["states"].append(state)
        self._st_memory["distributions"].append(distribution)

    def _store(self):
        print("MEMORY: Storing to long-term-memory")
        assert len(self._st_memory["states"]) == len(self._st_memory["distributions"]), "Must save equal states and dist"

        # Store short term memory in long term memory
        for i in range(len(self._st_memory["states"])):
            result = self._st_memory["result"]
            state = self._st_memory["states"][i]
            dist = self._st_memory["distributions"][i]

            self._lt_memory.append(
                [state, dist, result]
            )

        # Reset short term memory
        self._st_memory = {
            "result": None,
            "states": [],
            "distributions": [],
        }

    def sample(self):
        samples = random.sample(self._lt_memory, k=math.ceil(len(self._lt_memory) * self._sample_size))
        states, dists, results = zip(*samples)
        print(f"MEMORY: Returning {len(states)} samples")

        return np.array(states), {"value": np.array(results), "policy": np.array(dists)}

