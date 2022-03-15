class TOPP:
    def __init__(self):
        pass


if __name__ == "__main__":
    from actor import Actor
    from hex import HexGame, GameWindow
    from agents.buffer_agent import BufferAgent

    env = HexGame(size=4)
    act = Actor(
        input_size=len(env.flat_state),
        output_size=len(env.legal_binary_moves),
        hidden_size=(200, 100),
        learning_rate=0.001
    )
    act.load_saved_model("(1) S4_B25.h5")
    buffer = BufferAgent()
    window = GameWindow(width=1000, height=600, game=env.copy(), agent=buffer, view_update_rate=2.0)

    env.execute((0, 0))
    buffer.add_move((0, 0))

    while not env.is_game_over:
        move = act.best_move(env)
        env.execute(move)
        buffer.add_move(move)

    window.run()