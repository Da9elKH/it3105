import logging

from agents import CNNAgent
from config import App
from environments import Hex, HexGUI
from networks import CNN
from oht import OHT
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

    for task in App.config("run"):

        # REINFORCEMENT LEARNING
        if task == "RL":
            exec(open("rl.py").read())

        # TOURNAMENT OF PROGRESSIVE POLICIES
        elif task == "TOPP":
            topp = TOPP(environment=environment)
            matches = App.config("topp.matches")

            # TODO: Only add these when at TOPP
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
