from reinfin.config import Config
import reinfin.constants as const

from datetime import datetime


class BacktestConfig(Config):

    api_key: str
    api_secret: str
    base_url: str

    start_date: str
    end_date: str

    cash: int

    symbol: str

    def __init__(self, config):
        super().__init__(config)
        self.alpaca_config = {
            const.API_KEY_KEY: self.api_key,
            const.API_SECRET_KEY: self.api_secret,
            const.BASE_URL_KEY: self.base_url,
        }

        self.start_date = datetime.strptime(self.start_date, const.DATE_FORMAT_STR)
        self.end_date = datetime.strptime(self.end_date, const.DATE_FORMAT_STR)
