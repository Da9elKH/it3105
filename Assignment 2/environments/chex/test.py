# Python imports
import cProfile
import numpy
from topp import TOPP
from agents import MCTSAgent
from mcts import MCTS

# Cython imports
import pyximport
pyximport.install(
    setup_args={
        "include_dirs": [numpy.get_include()],
        "define_macros": [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    }
)
import modules

if __name__ == "__main__":

    env = modules.HexGame(size=7)

    # CCCC
    mcts1 = modules.MCTS(environment=env, time_budget=1, rollouts=1, c=1.5, verbose=True)
    agent1 = MCTSAgent(environment=env, model=mcts1)

    # PYTHON
    mcts2 = MCTS(environment=env, time_budget=1, rollouts=1, c=1.5, verbose=True, epsilon=1.0, rollout_policy_agent=None)
    agent2 = MCTSAgent(environment=env, model=mcts2)

    topp = TOPP(environment=env)
    topp.add_agent("cython", agent1)
    topp.add_agent("python", agent2)

