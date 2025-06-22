from reinfin.config import Config, ConfigError
import reinfin.constants as const

from alpaca_trade_api.common import URL

from datetime import datetime
import os


class TechIndicatorsConfig(Config):

    logging_path: str
    pipeline_id: str

    df_path: str

    ema_period: int

    rsi_period: int

    macd_period_fast: int
    macd_period_slow: int
    macd_signal: int

    bbands_period: int

    stoch_period: int

    def __init__(self, config):
        super().__init__(config)
        self.save_directory = getattr(
            self,
            "save_directory",
            f"data/{self.pipeline_id}",
        )
        # If save path doesn't already exist, create the directory (needed to make new pipeline-named directories)
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory, exist_ok=True)
        self.save_filename = "".join(
            [os.path.basename(self.df_path).removesuffix(".csv"), "_ti", ".csv"]
        )
        self.save_path = os.path.join(self.save_directory, self.save_filename)

    @property
    def required_config(self):
        return {
            "logging_path": str,
            "pipeline_id": str,
            "df_path": str,
            "ema_period": int,
            "rsi_period": int,
            "macd_period_fast": int,
            "macd_period_slow": int,
            "macd_signal": int,
            "bbands_period": int,
            "stoch_period": int,
        }
