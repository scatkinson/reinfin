from reinfin.processing.tech_indicators_config import TechIndicatorsConfig
import reinfin.constants as const

from finta import TA
import logging
import pandas as pd


class TechIndicators:
    def __init__(self, conf: TechIndicatorsConfig):
        self.conf = conf
        self.df = pd.read_csv(self.conf.df_path, index_col=const.TIMESTAMP_COL)

    def run_tech_indicators(self):
        self.obtain_indicators()
        self.save_out_df()

    def obtain_indicators(self):
        self.compute_ema()
        self.compute_rsi()
        self.compute_macd()
        self.compute_bbands()
        self.compute_stoch()
        self.compute_vwap()
        logging.info(f"Updated schema of DF: {self.df.columns}")

    def compute_ema(self):
        logging.info("Obtaining EMA values")
        self.df[const.EMA_COL] = TA.EMA(self.df, period=self.conf.ema_period)
        logging.info("Shifting EMA values")
        self.df[const.EMA_COL] = self.df[const.EMA_COL].shift(1)

    def compute_rsi(self):
        logging.info("Obtaining RSI values")
        self.df[const.RSI_COL] = TA.RSI(self.df, period=self.conf.rsi_period)
        logging.info("Shifting RSI values")
        self.df[const.RSI_COL] = self.df[const.RSI_COL].shift(1)

    def compute_macd(self):
        logging.info("Obtaining MACD values")
        self.df[[const.MACD_COL, const.MACD_SIGNAL_COL]] = TA.MACD(
            self.df,
            period_fast=self.conf.macd_period_fast,
            period_slow=self.conf.macd_period_slow,
            signal=self.conf.macd_signal,
        )
        logging.info("Shifting MACD values")
        self.df[[const.MACD_COL, const.MACD_SIGNAL_COL]] = self.df[
            [const.MACD_COL, const.MACD_SIGNAL_COL]
        ].shift(1)

    def compute_bbands(self):
        logging.info("Obtaining BBANDS values")
        self.df[
            [const.UPPER_BBANDS_COL, const.MIDDLE_BBANDS_COL, const.LOWER_BBANDS_COL]
        ] = TA.BBANDS(self.df, period=self.conf.bbands_period)
        logging.info("Shifting BBANDS values")
        self.df[
            [const.UPPER_BBANDS_COL, const.MIDDLE_BBANDS_COL, const.LOWER_BBANDS_COL]
        ] = self.df[
            [const.UPPER_BBANDS_COL, const.MIDDLE_BBANDS_COL, const.LOWER_BBANDS_COL]
        ].shift(
            1
        )

    def compute_stoch(self):
        logging.info("Obtaining STOCH values")
        self.df[const.STOCH_COL] = TA.STOCH(self.df, period=self.conf.stoch_period)
        logging.info("Shifting STOCH values")
        self.df[const.STOCH_COL] = self.df[const.STOCH_COL].shift(1)

    def compute_vwap(self):
        logging.info("Obtaining VWAP values")
        self.df[const.VWAP_COL] = TA.VWAP(self.df)
        logging.info("Shifting VWAP values")
        self.df[const.VWAP_COL] = self.df[const.VWAP_COL].shift(1)

    def save_out_df(self):
        logging.info(
            f"Saving updated DF with Technical Indicators at {self.conf.save_path}"
        )
        self.df.fillna(method="bfill", inplace=True)
        self.df.to_csv(self.conf.save_path)
