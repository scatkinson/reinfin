from reinfin.processing import TechIndicatorsConfig, TechIndicators
from reinfin.configamend import configamend

from reinfin import log_wu

import argparse
import logging
import yaml
import sys

parser = argparse.ArgumentParser(
    prog="technical_indicators",
    description="Appends technical indicators to raw stocks file",
    epilog="",
)

parser.add_argument(
    "-c",
    "--config",
    required=True,
    help="Path of config file, e.g. from working directory: bin/processing/conf/test_config.yml",
)

args = parser.parse_args()


def main(conf: TechIndicatorsConfig):
    ti = TechIndicators(conf)
    ti.run_tech_indicators()


if __name__ == "__main__":
    with open(args.config, mode="r") as f:
        config = configamend(yaml.safe_load(f))
    conf = TechIndicatorsConfig(config)
    attrs = vars(conf)
    logging.info("Namespace: " + ", ".join("%s: %s" % item for item in attrs.items()))

    logging.info(f"Running main() of {__file__}")
    logging.info(f"Log will appear at {conf.logfile}")

    log_wu.run_with_logging(main, conf, logfile=conf.logfile)
