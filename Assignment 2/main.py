import logging

from agents import CNNAgent, ANNAgent
from config import App
from environments import Hex, HexGUI
from networks import CNN, ANN
from oht import OHT
from rl import ReinforcementLearner
from topp import TOPP


def setup_logger():
    logging.basicConfig(
        filename="logs.log",
        filemode="a",
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        level=logging.DEBUG
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def run():
    environment = Hex(size=App.config("environment.size"))
    models = []

    for task in App.config("run"):
        # REINFORCEMENT LEARNING
        if task == "RL":
            print("RL STARTED")
            rl = ReinforcementLearner()
            rl.run()
            models = rl.saved_models

        # TOURNAMENT OF PROGRESSIVE POLICIES
        elif task == "TOPP":
            print("TOPP STARTED")
            topp = TOPP(environment=environment)
            matches = App.config("topp.matches")

            if models:
                for model in models:
                    agent, network = (CNNAgent, CNN) if App.config("rl.use_cnn") else (ANNAgent, ANN)
                    topp.add_agent(model, agent(environment=environment, network=network.from_file(model)))
            else:
                topp.add_agent("0", CNNAgent(environment=environment, network=CNN.from_file("5x5/(1) CNN_S5_B0.h5")))
                topp.add_agent("75", CNNAgent(environment=environment, network=CNN.from_file("5x5/(1) CNN_S5_B75.h5")))
                topp.add_agent("150", CNNAgent(environment=environment, network=CNN.from_file("5x5/(1) CNN_S5_B150.h5")))
                topp.add_agent("225", CNNAgent(environment=environment, network=CNN.from_file("5x5/(1) CNN_S5_B225.h5")))
                topp.add_agent("300", CNNAgent(environment=environment, network=CNN.from_file("5x5/(1) CNN_S5_B300.h5")))

            if App.config("topp.visualize"):
                gui = HexGUI(environment=environment)
                gui.run_visualization_loop(lambda: topp.tournament(matches))
            else:
                topp.tournament(matches)

        # ONLINE HEX TOURNAMENT
        elif task == "OHT":
            oht = OHT(auth=App.config("oht.auth"), qualify=App.config("oht.qualify"), environment=environment)

            if App.config("oht.visualize"):
                gui = HexGUI(environment=environment)
                gui.run_visualization_loop(lambda: oht.run(mode=App.config("oht.mode")))
            else:
                oht.run(mode=App.config("oht.mode"))


if __name__ == "__main__":
    setup_logger()
    run()
