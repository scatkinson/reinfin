from reinfin.agents.rl_tradingbot import RLTradingBot
from lumibot.brokers import Alpaca
from reinfin.agents.backtest_config import BacktestConfig
from reinfin.agents.rl_tradingbot_config import RLTradingbotConfig
import reinfin.constants as const
from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting


class BacktestRunner:
    def __init__(self, conf: BacktestConfig):
        self.conf = conf

        self.broker = Alpaca(self.conf.alpaca_config)
        rltb_config_dict = {
            const.SYMBOL_KEY: self.conf.symbol,
        }
        rltb_conf = RLTradingbotConfig(rltb_config_dict)
        self.strategy = RLTradingBot(self.broker, conf=rltb_conf)

    def run_backtest(self):
        self.strategy.backtest(
            YahooDataBacktesting,
            self.conf.start_date,
            self.conf.end_date,
            # cash=self.conf.cash,
        )
