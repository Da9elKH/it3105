from utils.types import GameType, CriticType
from utils.decaying_variable import DecayingVariable
from actor import Actor
from agent import Agent
from critics.nn import NeuralNetworkCritic
from critics.table import TableCritic
from environments.the_gambler import TheGambler
from environments.cart_pole import CartPole
from environments.tower_of_hanoi import TowerOfHanoi
from environments.gym_environment import AIGym


if __name__ == '__main__':
    game = GameType.CART_POLE
    critic = CriticType.TABLE

    if game == GameType.CART_POLE:
        environment = CartPole(
            pole_length=0.5,
            pole_mass=0.1,
            gravity=-9.8,
            time_step=0.02,
            buckets=(5, 5, 5, 5),
            view_update_rate=0.02,
            time_out=300
        )

        if critic == critic.NEURAL_NETWORK:
            n_episodes = 300
            critic = NeuralNetworkCritic(
                discount_factor=0.9,
                learning_rate=0.003,
                input_size=environment.input_space(),
                hidden_size=(32, 32)
            )
        else:
            n_episodes = 500
            critic = TableCritic(
                discount_factor=0.9,
                trace_decay=0.9,
                learning_rate=0.2
            )
        actor = Actor(
            discount_factor=0.9,
            trace_decay=0.9,
            learning_rate=0.2,
            epsilon=DecayingVariable(
                start=1.0,
                end=0.005,
                linear=True,
                episodes=n_episodes,
                episodes_end_value=0
            ),
        )
    elif game == GameType.TOWER_OF_HANOI:
        n_episodes = 500
        environment = TowerOfHanoi(
            num_pegs=3,
            num_discs=4,
            view_update_rate=0.5,
            time_out=300
        )
        if critic == CriticType.NEURAL_NETWORK:
            critic = NeuralNetworkCritic(
                discount_factor=0.95,
                learning_rate=0.003,
                input_size=environment.input_space(),
                hidden_size=(64, 32)
            )
        else:
            critic = TableCritic(
                discount_factor=0.9,
                trace_decay=0.6,
                learning_rate=0.05
        )
        actor = Actor(
            discount_factor=0.9,
            trace_decay=0.6,
            learning_rate=0.01,
            epsilon=DecayingVariable(
                start=1.0,
                end=0.001,
                episodes=n_episodes - 25,
                linear=False
            )
        )
    elif game == GameType.GAMBLER:
        environment = TheGambler(
            win_probability=0.4,
            state_space=100,
            time_out=300
        )
        if critic == CriticType.NEURAL_NETWORK:
            n_episodes = 1000
            critic = NeuralNetworkCritic(
                discount_factor=1,
                learning_rate=0.003,
                input_size=environment.input_space(),
                hidden_size=(64, 32, 16)
            )
        else:
            n_episodes = 100000
            critic = TableCritic(
                discount_factor=1,
                trace_decay=0.8,
                learning_rate=0.03
            )
        actor = Actor(
            discount_factor=1,
            trace_decay=0.8,
            learning_rate=0.01,
            epsilon=DecayingVariable(
                start=1,
                end=0.001,
                episodes=n_episodes,
                linear=False
            ),
        )
    elif game == GameType.AI_GYM:
        environment = AIGym("CartPole-v1", state_shape=(11, 11, 11, 11))
        if critic == critic.NEURAL_NETWORK:
            n_episodes = 300
            critic = NeuralNetworkCritic(
                discount_factor=0.9,
                learning_rate=0.003,
                input_size=environment.input_space(),
                hidden_size=(32, 32)
            )
        else:
            n_episodes = 5000
            critic = TableCritic(
                discount_factor=0.9,
                trace_decay=0.9,
                learning_rate=0.2
            )
        actor = Actor(
            discount_factor=0.9,
            trace_decay=0.9,
            learning_rate=0.2,
            epsilon=DecayingVariable(
                start=1.0,
                end=0.005,
                linear=False,
                episodes=n_episodes,
                episodes_end_value=0
            ),
        )
    else:
        raise ValueError("Select a game environment")

    agent = Agent(env=environment, act=actor, crt=critic)
    agent.actor_critic_model(n_episodes)
