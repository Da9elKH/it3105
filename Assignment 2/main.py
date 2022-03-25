from agents.mcts_agent import MCTSAgent
from environments.hex import HexGame
from environments.hex_gui import HexGUI
from agents.buffer_agent import BufferAgent
from agents.cnn_agent import CNNAgent
import time
import numpy as np
from collections import deque
from tqdm import trange
from networks.cnn import CNN
from mcts.mcts import MCTS
from memory import Memory

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


if __name__ == "__main__" and False:
    env = HexGame(size=5)
    cnn = CNN.build(
        input_size=env.cnn_state.shape,
        output_size=len(env.legal_binary_moves),
        hidden_size=(100, 100),
        learning_rate=0.01
    )

    env.execute(env.legal_moves[1])
    policy, value = cnn.predict(np.array([env.cnn_state]))

    state = env.cnn_state

    states = [env.cnn_state, env.cnn_state]
    dist = [
        [0.06736019058271145, 0.048737338572804004, 0.0758244133619637, 0.064406319523012, 0.07786098326732228, 0.054453638230643815, 0.011950307165072085, 0.05674136502540529, 0.03527505283997416, 0.05869097228648447, 0.02973925262513724, 0.04209247281068154, 0.03181785991331279, 0.0679823289288545, 0.00018903287823166247, 0.022998280586276603, 0.017706474868363603, 0.03384514142812192, 0.002768011876237949, 0.03796881174811376, 0.007427062489231768, 0.002128655885188717, 0.07356309227874756, 0.02419229115442113, 0.05428064967368612]
        , [0.001909641459584813, 0.06055160228915245, 0.008632949066138033, 0.08624008897874953, 0.025894826629573233, 0.07299329776663183, 0.008800718575675435, 0.07488051560683318, 0.019423718714081464, 0.07509161869906991, 0.004944997531498252, 0.07612576516903723, 0.024803566355057902, 0.04724836047005981, 0.06625387899860705, 0.009446565536546627, 0.029503434303820675, 0.042712339425585895, 0.00690226564279825, 0.05396573522690465, 0.007409939032288306, 0.08650837771960838, 0.01722514848928221, 0.053280827174180756, 0.039249821139234066]
    ]
    res = [-1, 1]

    x = np.array(states)
    y = {'value': np.array(res), 'policy': np.array(dist)}

    print(cnn.train_on_batch(x, y))


if __name__ == "__main__":

    env = HexGame(size=7, start_player=-1)
    network = CNN.build(
        input_size=env.cnn_state.shape,
        output_size=len(env.legal_binary_moves),
        hidden_size=(100, 100), # TODO: DOESNT DO ANYTHING
        learning_rate=0.1
    )
    mcts = MCTS(
        rollout_policy_agent=CNNAgent(network=network),
        environment=env,
        time_budget=1.0,
        #rollouts=2500,
        epsilon=1.00,
        verbose=True,
        c=1.0  # 1 til 3
    )
    agent = MCTSAgent(
        environment=env,
        model=mcts,
    )

    """ RENDERING """
    buffer = BufferAgent(environment=env)
    window = HexGUI(width=1000, height=600, game=env.copy(), agent=buffer, view_update_rate=2.0)
    memory = Memory(sample_size=0.25, queue_size=128)

    BATCHES = 100
    with trange(BATCHES) as t:
        t.set_description("Training..")

        for i in t:
            mcts.reset()
            env.reset()

            while not env.is_game_over:
                # Run MCTS
                best_move, distribution = agent.get_move(greedy=True)

                # Format distribution to be viewed in game board
                meta_dist = {index: str(round(v * 100, 2)) + "%" for index, v in np.ndenumerate(distribution.reshape(env.state.shape))}

                # Buffer for replay
                buffer.add_distribution(meta_dist)
                buffer.add_move(best_move)

                # Add to buffer
                memory.register_state_and_distribution(env.cnn_state, distribution)

                # Change environment
                env.play(best_move)
                mcts.move(best_move)

            window.run()
            memory.register_result(env.result)

            # Train actor on game played
            inputs, targets = memory.sample()
            network.train_on_batch(inputs, targets)

            if i % 10 == 0:
                model_name = lambda size, batch: f"S{size}_B{i}"
                network.save_model(suffix=model_name(env.size, BATCHES))

if __name__ == "__main__" and False:
    env = HexGame(size=7)
    env.current_player = 2
    window = HexGUI(width=1000, height=600, game=env.copy(), agent=None, view_update_rate=2.0)
    window.run()


if __name__ == "__main__" and False:
    QUEUE_SIZE = 128
    batches = 2000

    env = HexGame(size=7)
    """
    network = ANN.build(
        input_size=len(env.flat_state),
        output_size=len(env.legal_binary_moves),
        hidden_size=(200, 100),
        learning_rate=0.001
    )
    """
    network = CNN.build(
        input_size=env.cnn_state.shape,
        output_size=len(env.legal_binary_moves),
        hidden_size=(100, 100),
        learning_rate=0.1
    )
    mcts = MCTS(
        rollout_policy_agent=CNNAgent(network=network),
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
    window = HexGUI(width=1000, height=600, game=env.copy(), agent=buffer, view_update_rate=2.0)

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
                RBUF_input.append(mcts.environment.cnn_state)
                RBUF_target.append(distribution)

                # Change environment
                env.execute(best_move)
                mcts.move(best_move)

            # Train actor on game played
            network.train_on_batch(RBUF_input, RBUF_target)
            window.run()

            if i % 25 == 0:
                model_name = lambda size, batch: f"S{size}_B{i}"
                network.save_model(suffix=model_name(env.size, batches))

    #window.run()
