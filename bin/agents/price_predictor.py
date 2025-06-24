from reinfin.agents import PricePredictorConfig, PricePredictor
from reinfin.configamend import configamend

from reinfin import log_wu

import argparse
import logging
import yaml
import sys

parser = argparse.ArgumentParser(
    prog="ddqn_runner",
    description="Runs price prediction",
    epilog="",
)

parser.add_argument(
    "-c",
    "--config",
    required=True,
    help="Path of config file, e.g. from working directory: bin/agents/conf/price_predictor_test_config.yml",
)

args = parser.parse_args()


def main(conf: PricePredictorConfig):
    pp = PricePredictor(conf)
    pp.run_price_predictor()


if __name__ == "__main__":
    with open(args.config, mode="r") as f:
        config = configamend(yaml.safe_load(f))
    conf = PricePredictorConfig(config)
    attrs = vars(conf)
    logging.info("Namespace: " + ", ".join("%s: %s" % item for item in attrs.items()))

    logging.info(f"Running main() of {__file__}")
    logging.info(f"Log will appear at {conf.logfile}")

    log_wu.run_with_logging(main, conf, logfile=conf.logfile)
