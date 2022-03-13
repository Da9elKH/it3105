from hex import HexGame
from mcts import Node
from actor import Actor


if __name__ == "__main__":
    #game = HexGame(size=3)
    #agent = MCTSAgent()
    #rendering = GameWindow(1000, 750, game=game, agent=agent)
    #rendering.run()

    """ TESTING A COMPLETE ALGORITHM """
    epocs = 1

    game = HexGame(size=4)
    mc_game = HexGame(size=4)
    actor = Actor(
        learning_rate=0.03,
        input_size=len(game.flat_state),
        output_size=len(game.legal_moves_binary()),
        hidden_size=(128, 64, 32)
    )

    for i in range(epocs):
        game.reset()
        mc_game.reset()
        node = Node(mc_game, actor=actor)

        while not game.is_game_over:
            new_node = node.best_action(use_time=True, time_limit=2.0)
            game.execute(new_node.parent_action)

