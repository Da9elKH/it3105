from environments.cart_pole import CartPole
from environments.tower_of_hanoi import TowerOfHanoi
from environments.the_gambler import TheGambler
from environments.environment import ProblemEnvironment
from critics.critic import Critic
from critics.nn import NeuralNetworkCritic
from utils.decaying_variable import DecayingVariable
from critics.table import TableCritic
from actor import Actor


class Agent:
    def __init__(self, env: ProblemEnvironment, act: Actor, crt: Critic):
        self.env = env
        self.act = act
        self.crt = crt

    def actor_critic_model(self, episodes: int):
        win = 0

        for episode in range(episodes):
            # Reset eligibilities for actor and critic
            self.crt.clear()
            self.act.clear()

            # Initialize environment
            state, valid_actions = self.env.reset()

            # Helpers for replay logic
            last_episode = episode == (episodes - 1)
            i = 0

            # Store data for later visualisation
            self.env.store_training_metadata(current_episode=episode, last_episode=last_episode, current_step=i, state=state)

            while not self.env.is_finished():
                # ACTOR: Get next action
                action = self.act.next_action(state, valid_actions)

                # ENVIRONMENT: Do next action and receive reinforcement, save state in list
                from_state, action, reinforcement, state, valid_actions, done = self.env.step(action)

                # ACTOR: Get next action and update eligibility
                self.act.set_eligibility(from_state, action)

                # CRITIC: Calculate TD-error and update eligibility
                td_error = self.crt.td_error(reinforcement, from_state, state, done)
                self.crt.set_eligibility(from_state)

                # Adjustments for all state-action-pairs
                self.crt.adjust(td_error)
                self.act.adjust(td_error)

                # Store data for later visualisation
                i += 1
                self.env.store_training_metadata(current_episode=episode, last_episode=last_episode, current_step=i, state=state)

            self.crt.learn()

            if self.env.has_succeeded():
                win += 1
            print(f"Episode: {episode}, steps: {i}, win: {self.env.has_succeeded()}")
        print(f"Wins: {win}")
        self.env.replay(saps=self.act.get_saps(), values=self.crt.get_values())

if __name__ == '__main__':

    """ CARTPOLE """
    n_episodes = 300  # 1000
    environment = CartPole(pole_length=0.5, pole_mass=0.1, gravity=-9.8, timestep=0.02, buckets=(5, 5, 6, 6))
    actor = Actor(
        discount_factor=0.2,
        trace_decay=0.8,
        #learning_rate=DecayingVariable(
        #    start=0.4,
        #    end=0.001,
        #    episodes=n_episodes,
        #),
        learning_rate=0.1,
        epsilon=DecayingVariable(
            start=1.0,
            end=0.01,
            #decay=0.995,
            episodes=n_episodes,
        ),
    )

    critic = NeuralNetworkCritic(
        discount_factor=0.7,
        learning_rate=0.001,
        input_size=environment.input_space(),
        hidden_size=(32, 32)
    )
    """
    critic = TableCritic(
        discount_factor=0.7,
        trace_decay=0.6,
        learning_rate=DecayingVariable(
            start=0.6,
            end=0.001,
            episodes=n_episodes,
        )
    )
    """


    """ TOWER OF HANOI
    n_episodes = 500
    environment = TowerOfHanoi(num_pegs=3, num_discs=4)
    actor = Actor(
        discount_factor=0.95,
        trace_decay=0.5,
        learning_rate=0.4,
        epsilon=DecayingVariable(
            start=1,
            end=0.01,
            episodes=n_episodes-25,
            linear=False
        )
    )
    critic = NeuralNetworkCritic(
        discount_factor=0.95,
        learning_rate=0.03,
        input_size=environment.input_space(),
        hidden_size=(12, 12, 12)
    )
    critic = TableCritic(discount_factor=0.95, trace_decay=0.5, learning_rate=0.1)
    """

    """ THE GAMBLER
    n_episodes = 3000
    environment = TheGambler(win_probability=0.4, state_space=100)
    actor = Actor(
        discount_factor=1,
        trace_decay=0.2,
        learning_rate=0.4,
        epsilon=DecayingVariable(
            start=1,
            end=0.01,
            episodes=n_episodes,
            linear=False
        ),
    )
    critic = NeuralNetworkCritic(
        discount_factor=1,
        learning_rate=0.01,
        input_size=environment.input_space(),
        hidden_size=(32, 32)
    )
    critic = TableCritic(discount_factor=1, trace_decay=0.2, learning_rate=0.01)
    """

    """ Training """
    agent = Agent(env=environment, act=actor, crt=critic)
    agent.actor_critic_model(n_episodes)
