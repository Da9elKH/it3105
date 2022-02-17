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
        """ This functions runs the general actor critic model """

        # Initializing small state-values for CRITIC is done when accessing the value
        # Initializing 0 as SAP-value for ACTOR is done when accessing the value

        for episode in range(episodes):
            # Reset eligibilities for actor and critic
            # e(s,a) ← 0: e(s) ← 0 ∀s, a
            self.crt.clear()
            self.act.clear()

            # Initialize environment
            # s ← s_init
            state, valid_actions = self.env.reset()

            # Helpers for replay logic
            last_episode = episode == (episodes - 1)
            i = 0

            # Store data for later visualisation
            self.env.store_training_metadata(current_episode=episode, last_episode=last_episode, current_step=i, state=state, reinforcement=0)

            while not self.env.is_finished():
                # Keep track of the current state-step
                i += 1

                # ACTOR: Get next action
                # a′ ← Π(s′)
                action = self.act.next_action(state, valid_actions)

                # ENVIRONMENT: Do next action and receive reinforcement, save state in list
                from_state, action, reinforcement, state, valid_actions, terminal = self.env.step(action)

                # ACTOR: Get next action and update eligibility
                # e(s,a) ← 1
                self.act.set_eligibility(from_state, action)

                # CRITIC: Calculate TD-error and update eligibility
                # δ ← r+γV(s′)−V(s)
                td_error = self.crt.td_error(reinforcement, from_state, state, terminal)
                # e(s) ← 1
                self.crt.set_eligibility(from_state)

                # Adjustments for all state-action-pairs
                # ∀(s,a):
                #   V(s) ← V(s)+ α*δ*e(s)
                #   e(s) ← γλe(s)
                #   Π(s,a) ← Π(s,a)+α*δ*e(s,a)
                #   e(s,a) ← γλe(s,a)
                self.crt.adjust(td_error)
                self.act.adjust(td_error)

                # Store data for later visualisation
                self.env.store_training_metadata(current_episode=episode, last_episode=last_episode, current_step=i, state=state, reinforcement=reinforcement)

            # For batch learning if using neural network
            loss = self.crt.learn()
            print(f"Episode: {episode}, steps: {i}, loss: {loss}")

        self.env.replay(saps=self.act.get_saps(), values=self.crt.get_values())
