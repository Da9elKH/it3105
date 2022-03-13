from agent import Agent


class BufferAgent(Agent):
    def __init__(self):
        self.action_buffer = []
        self.reversed = False

    def action(self, state, valid_actions, game):
        if not self.reversed:
            self.action_buffer.reverse()
            self.reversed = True

        if self.action_buffer:
            return self.action_buffer.pop()
        else:
            return None

    def add_to_buffer(self, action):
        self.action_buffer.append(action)
