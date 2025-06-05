from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
import pandas as pd
from reinfin.extract.extractor_config import ExtractorConfig
import logging


class Extractor:
    def __init__(self, conf: ExtractorConfig):
        self.conf = conf
        self.api = REST(self.conf.api_key, self.conf.api_secret, self.conf.base_url)
        self.timeframe = TimeFrame(1, TimeFrameUnit.Day)
        if self.conf.timeframe_str == "Hour":
            self.timeframe = TimeFrame(1, TimeFrameUnit.Hour)

    def get_data(self):
        logging.info("Obtaining data")
        barset = self.api.get_bars(
            self.conf.symbol,
            timeframe=self.timeframe,
            start=self.conf.start_date,
            end=self.conf.end_date,
        ).df
        logging.info(f"Saving data to {self.conf.save_path}.")
        barset.to_csv(self.conf.save_path)
