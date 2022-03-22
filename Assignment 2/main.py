from agents.mcts_agent import MCTSAgent
from hex import HexGame, GameWindow
from actor import Actor
from agents.buffer_agent import BufferAgent
from agents.ann_agent import ANNAgent
import time
import numpy as np
from collections import deque
from tqdm import tqdm, trange
from ann import Network
from mcts.mcts import MCTS

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'



if __name__ == "__main__":
    env = HexGame(size=7)
    env.current_player = 2
    window = GameWindow(width=1000, height=600, game=env.copy(), agent=None, view_update_rate=2.0)
    window.run()


if __name__ == "__main__" and False:

    QUEUE_SIZE = 128
    batches = 2000

    env = HexGame(size=7)
    ann = Network.build(
        input_size=len(env.flat_state),
        output_size=len(env.legal_binary_moves),
        hidden_size=(200, 100),
        learning_rate=0.001
    )
    mcts = MCTS(
        rollout_policy_agent=ANNAgent(network=ann),
        environment=env,
        time_budget=1.0,
        c=1.0, # 1 til 3
        epsilon=0.0
    )
    agent = MCTSAgent(
        environment=env,
        model=mcts,
    )

    """ RENDERING """
    buffer = BufferAgent()
    window = GameWindow(width=1000, height=600, game=env.copy(), agent=buffer, view_update_rate=2.0)

    RBUF_input = deque([], QUEUE_SIZE)
    RBUF_target = deque([], QUEUE_SIZE)

    with trange(batches) as t:
        t.set_description("Training..")

        for i in t:
            mcts.reset()
            env.reset()

            best_move = None
            start_time = time.time()

            while not env.is_game_over:
                # Run MCTS
                best_move, distribution = agent.get_move(greedy=True)

                # Format distribution to be viewed in game board
                meta_dist = {index: str(round(v * 100, 2)) + "%" for index, v in np.ndenumerate(distribution.reshape(env.state.shape))}

                # Buffer for replay
                buffer.add_distribution(meta_dist)
                buffer.add_move(best_move)

                # Add to buffer
                RBUF_input.append(mcts.environment.flat_state)
                RBUF_target.append(distribution)

                # Change environment
                env.execute(best_move)
                mcts.move(best_move)

            # Train actor on game played
            ann.train_on_batch(RBUF_input, RBUF_target)
            window.run()

            if i % 25 == 0:
                model_name = lambda size, batch: f"S{size}_B{i}"
                ann.save_model(suffix=model_name(env.size, batches))

    #window.run()
