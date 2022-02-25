from environments.hex import HexGame, GameWindow

if __name__ == "__main__":
    game = HexGame(size=10)
    rendering = GameWindow(500, 750)
    rendering.setup(game)
    rendering.run()
