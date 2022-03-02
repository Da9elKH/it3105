from hex import HexGame, GameWindow
from agents.random_agent import RandomAgent


if __name__ == "__main__":
    game = HexGame(size=5)
    agent = RandomAgent()
    rendering = GameWindow(1000, 750, game=game, agent=agent)
    rendering.run()
