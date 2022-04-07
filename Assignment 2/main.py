import logging
import colorlog
from agents import CNNAgent, ANNAgent
from config import App
from environments import Hex, HexGUI
from networks import CNN, ANN, Network
from oht import OHT
from rl import ReinforcementLearner
from topp import TOPP

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logger(save_logs=False):
    if save_logs:
        logging.basicConfig(
            filename="logs.log",
            filemode="a",
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            level=logging.DEBUG
        )

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)-7s%(reset)s%(name_log_color)s%(name)-12s %(message_log_color)s%(message)s',
            secondary_log_colors={
                "name": {
                    'DEBUG':    'cyan',
                    'INFO':     'cyan',
                    'WARNING':  'cyan',
                    'ERROR':    'cyan',
                    'CRITICAL': 'cyan',
                },
                "message": {
                    'DEBUG':    'white',
                    'INFO':     'white',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'red,bg_white',
                }
            },
            style='%'
        )
    )
    colorlog.getLogger('').addHandler(handler)


def run():
    environment = Hex(size=App.config("environment.size"))
    models = App.config("topp.models")

    for task in App.config("run"):
        # REINFORCEMENT LEARNING
        if task == "RL":
            logger.info("Starting Reinforcement Learning")
            rl = ReinforcementLearner()
            rl.run()
            models = rl.saved_models

        # TOURNAMENT OF PROGRESSIVE POLICIES
        elif task == "TOPP":
            logger.info("Starting Tournament of Progressive Policies")
            topp = TOPP(environment=environment)
            matches = App.config("topp.matches")

            for filename in models:
                info = Network.info_from_path(filename)
                agent, network = (CNNAgent, CNN) if info.type == "CNN" else (ANNAgent, ANN)
                topp.add_agent(info.name, agent(environment=environment, network=network.from_file(filename)))

            if models:
                if App.config("topp.visualize"):
                    gui = HexGUI(environment=environment)
                    gui.run_visualization_loop(lambda: topp.tournament(matches))
                else:
                    topp.tournament(matches)

        # ONLINE HEX TOURNAMENT
        elif task == "OHT":
            logger.info("Starting Online Hex Tournament")
            filename = App.config("oht.model")
            info = Network.info_from_path(filename)
            agent, network = (CNNAgent, CNN) if info.type == "CNN" else (ANNAgent, ANN)

            oht = OHT(
                auth=App.config("oht.auth"),
                qualify=App.config("oht.qualify"),
                environment=environment,
                agent=agent(environment=environment, network=network.from_file(filename))
            )

            if App.config("oht.visualize"):
                gui = HexGUI(environment=environment)
                gui.run_visualization_loop(lambda: oht.run(mode=App.config("oht.mode")))
            else:
                oht.run(mode=App.config("oht.mode"))


if __name__ == "__main__":
    setup_logger(save_logs=False)
    run()
