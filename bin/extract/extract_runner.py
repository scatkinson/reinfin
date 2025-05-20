from reinfin.extract import Extractor, ExtractorConfig
from reinfin.configamend import configamend

from reinfin import log_wu

import argparse
import logging
import yaml
import sys

parser = argparse.ArgumentParser(
    prog="extract_runner",
    description="Runs data extract of trade data according to config",
    epilog="",
)

parser.add_argument(
    "-c",
    "--config",
    required=True,
    help="Path of config file, e.g. from working directory: bin/extract/conf/test_config.yml",
)

args = parser.parse_args()


def main(conf: ExtractorConfig):
    extractor = Extractor(conf)
    extractor.get_data()


if __name__ == "__main__":
    with open(args.config, mode="r") as f:
        config = configamend(yaml.safe_load(f))
    conf = ExtractorConfig(config)
    attrs = vars(conf)
    logging.info("Namespace: " + ", ".join("%s: %s" % item for item in attrs.items()))

    logging.info(f"Running main() of {__file__}")
    logging.info(f"Log will appear at {conf.logfile}")

    log_wu.run_with_logging(main, conf, logfile=conf.logfile)
