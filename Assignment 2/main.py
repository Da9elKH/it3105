from mcts import MCTSAgent
from hex import HexGame, GameWindow
from actor import Actor
from agents.buffer_agent import BufferAgent
import time
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    batches = 25
    #buffer = BufferAgent()

    env = HexGame(size=4)
    act = Actor(
        input_size=len(env.flat_state),
        output_size=len(env.legal_binary_moves),
        hidden_size=(200, 100),
        learning_rate=0.001
    )
    mcts = MCTSAgent(environment=env, actor=act)
    #window = GameWindow(width=1000, height=600, game=env.copy(), agent=buffer, view_update_rate=2.0)

    RBUF_input = []
    RBUF_target = []

    for i in tqdm(range(batches)):
        mcts.reset()
        env.reset()

        best_move = None
        start_time = time.time()

        while not env.is_game_over:
            # Run MCTS
            distribution, best_move = mcts.search(time_budget=2.0, epsilon=0.5, c=0.9)

            # Format distribution to be viewed in game board
            #meta_dist = {index: str(round(v * 100, 2)) + "%" for index, v in np.ndenumerate(distribution.reshape(env.state.shape))}

            # Buffer for replay
            #buffer.add_distribution(meta_dist)
            #buffer.add_move(best_move)

            # Add to buffer
            RBUF_input.append(mcts.environment.flat_state)
            RBUF_target.append(distribution)

            # Change environment
            env.execute(best_move)
            mcts.move(best_move)

        # Train actor on game played
        act.train(RBUF_input, RBUF_target)

    model_name = lambda size, batch: f"S{size}_B{batch}"
    act.save_model(suffix=model_name(env.size, batches))

    #window.run()
