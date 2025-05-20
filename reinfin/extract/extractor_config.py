from reinfin.config import Config, ConfigError
import reinfin.constants as const

from alpaca_trade_api.common import URL

from datetime import datetime
import os


class ExtractorConfig(Config):

    logging_path: str
    pipeline_id: str

    api_key: str
    api_secret: str
    base_url: URL

    start_date: str
    end_date: str

    timeframe_str: str

    symbol: str

    def __init__(self, config):
        super().__init__(config)
        self.alpaca_config = {
            const.API_KEY_KEY: self.api_key,
            const.API_SECRET_KEY: self.api_secret,
            const.BASE_URL_KEY: self.base_url,
        }
        if self.timeframe_str not in ["Day", "Hour"]:
            m = "timeframe_str must either be Day or Hour"
            raise ConfigError(m)
            # Save directory is the passed value if given, or the default value if none
        self.save_directory = getattr(
            self,
            "save_directory",
            f"data/{self.pipeline_id}",
        )
        # If save path doesn't already exist, create the directory (needed to make new pipeline-named directories)
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory, exist_ok=True)
        self.save_filename = (
            "_".join(
                [self.symbol]
                + [self.start_date]
                + [self.end_date]
                + [self.timeframe_str]
                + [self.pipeline_id]
            )
            + ".csv"
        )
        self.save_path = "/".join([self.save_directory, self.save_filename])

    @property
    def required_config(self):
        return {
            "logging_path": str,
            "pipeline_id": str,
            "api_key": str,
            "api_secret": str,
            "base_url": str,
            "start_date": str,
            "end_date": str,
            "timeframe_str": str,
            "symbol": str,
        }
