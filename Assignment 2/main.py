from environments.hex import HexGame, GameWindow
from agents.random_agent import RandomAgent
import cProfile

if __name__ == "__main__":
    game = HexGame(size=7)
    rendering = GameWindow(1000, 750)
    rendering.setup(game, agent=RandomAgent())
    rendering.run()
