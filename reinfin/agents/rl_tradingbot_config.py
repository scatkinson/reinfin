from reinfin.config import Config
import reinfin.constants as const

from datetime import datetime


class RLTradingbotConfig(Config):

    symbol: str

    def __init__(self, config):
        super().__init__(config)
