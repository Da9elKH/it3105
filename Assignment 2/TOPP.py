from agents.agent import Agent
from agents.ann_agent import ANNAgent
from agents.buffer_agent import BufferAgent
from ann import Network
from hex import HexGame, GameWindow


class TOPP:
    def __init__(self, environment: HexGame):
        self.environment = environment

    def tournament(self, player: Agent, opponent: Agent, rounds=50):
        stats = {}
        for i in range(rounds):
            print(f"Playing round {i}")
            result, winner = self.run_game(player, opponent)
            print(f"--> Winner: {'Opponent' if winner == 2 else 'Player'}")
            stats[i] = {"result": result, "winner": winner}

        print(stats)

    def run_game(self, player: Agent, opponent: Agent):
        buffer = BufferAgent()
        window = GameWindow(width=1000, height=600, game=self.environment.copy(), agent=buffer, view_update_rate=2.0)

        while not self.environment.is_game_over:
            if self.environment.current_player == 1:
                move = player.get_move(greedy=True)
            else:
                move = opponent.get_move(greedy=True)
            self.environment.execute(move)
            buffer.add_move(move)

        window.run()

        return self.environment.result, self.environment.current_player


if __name__ == "__main__":
    env = HexGame(size=4)
    topp = TOPP(environment=env)
    player = ANNAgent(environment=env, network=Network.from_file("(1) S4_B90.h5"))
    opponent = ANNAgent(environment=env, network=Network.from_file("(1) S4_B0.h5"))
    topp.tournament(player, opponent, 100)