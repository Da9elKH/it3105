from config import App
import logging
import random


def defaults():
    pass
    #random.seed(1)


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
    for task in App.config("run"):
        if task == "RL":
            exec(open("rl.py").read())
        elif task == "TOPP":
            exec(open("topp.py").read())
        elif task == "OHT":
            exec(open("oht.py").read())


if __name__ == "__main__":
    defaults()
    setup_logger()
    run()
