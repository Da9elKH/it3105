from .agent import Agent


class BufferAgent(Agent):
    def __init__(self):
        self.action_buffer = []
        self.distribution_buffer = []
        self.reversed = False
        self.first_pop = True

    def action(self, state, valid_actions, game):
        if not self.reversed:
            self.action_buffer.reverse()
            self.distribution_buffer.reverse()
            self.reversed = True

        if self.first_pop and self.distribution_buffer:
            self.first_pop = False
            return None, self.distribution_buffer.pop()
        elif not self.action_buffer and self.distribution_buffer:
            return None, self.distribution_buffer.pop()
        elif self.action_buffer and not self.distribution_buffer:
            return self.action_buffer.pop(), None
        elif self.action_buffer and self.distribution_buffer:
            return self.action_buffer.pop(), self.distribution_buffer.pop()
        else:
            return None, None

    def add_move(self, action):
        self.action_buffer.append(action)

    def add_distribution(self, distribution):
        self.distribution_buffer.append(distribution)
