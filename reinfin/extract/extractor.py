from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
import pandas as pd
from reinfin.extract.extractor_config import ExtractorConfig
import reinfin.constants as const
from reinfin.finbert_utils import estimate_sentiment
from timedelta import Timedelta
import logging


class Extractor:
    def __init__(self, conf: ExtractorConfig):
        self.conf = conf
        self.api = REST(self.conf.api_key, self.conf.api_secret, self.conf.base_url)
        self.timeframe = TimeFrame(1, TimeFrameUnit.Day)
        if self.conf.timeframe_str == "Hour":
            self.timeframe = TimeFrame(1, TimeFrameUnit.Hour)
        self.out_df = pd.DataFrame()

    def get_data(self):
        self.get_barsets()
        self.append_yesterday_close()
        self.get_sentiment()
        self.save_data()

    def get_barsets(self):
        logging.info("Obtaining barset data")
        self.out_df = self.api.get_bars(
            self.conf.symbol,
            timeframe=self.timeframe,
            start=self.conf.start_date,
            end=self.conf.end_date,
        ).df

    def append_yesterday_close(self):
        logging.info("Obtaining yesterday_close via df.shift(1).")
        self.out_df["yesterday_close"] = self.out_df["today_close"].shift(1)

    def compute_sentiment(self, date):
        prev_day = date - Timedelta(days=1)
        news = self.api.get_news(
            self.conf.symbol,
            start=prev_day.strftime(const.DATE_FORMAT_STR),
            end=date.strftime(const.DATE_FORMAT_STR),
        )
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        if sentiment == const.POS_STR:
            return probability.item()
        elif sentiment == const.NEG_STR:
            return -probability.item()
        else:
            return 0.0

    def get_sentiment(self):
        logging.info("Obtaining sentiment")
        for idx in self.out_df.index:
            date = idx.date()
            self.out_df.loc[idx, const.SENTIMENT_COL] = self.compute_sentiment(date)

    def save_data(self):
        logging.info(f"Saving data to {self.conf.save_path}.")
        self.out_df.to_csv(self.conf.save_path)
