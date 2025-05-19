from lumibot.brokers import Alpaca
from lumibot.strategies.strategy import Strategy
from reinfin.agents.rl_agent import RLAgent
from reinfin.agents.rl_tradingbot_config import RLTradingbotConfig
from typing import Optional
import pandas as pd
import numpy as np


class RLTradingBot(Strategy):

    def __init__(self, *args, conf: Optional[RLTradingbotConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = conf

    def initialize(self):
        self.symbol = (
            "SPY"  # TODO: set this as a config option that gets passed from config file
        )
        self.agent = RLAgent()
        self.last_state = None
        self.last_action = None

    def get_state(self, prices):
        # Normalize OHLCV and shares held
        recent = prices[-1]
        num_shares = 0
        if self.get_position(self.symbol):
            num_shares = self.get_position(self.symbol).quantity
        return (
            np.array(
                [
                    recent["open"],
                    recent["high"],
                    recent["low"],
                    recent["close"],
                    recent["volume"],
                    num_shares,
                ]
            )
            / 1000.0
        )

    def on_trading_iteration(self):
        hist = self.get_historical_prices(self.symbol, 60, "day")
        if hist is None or len(hist.df) < 1:
            return

        prices = hist.df.tail(1).to_dict("records")
        state = self.get_state(prices)

        if self.last_state is not None:
            current_price = prices[0]["close"]
            reward = self.get_portfolio_value()  # you can improve this
            self.agent.remember(self.last_state, self.last_action, reward, state, False)
            self.agent.replay()

        action = self.agent.act(state)
        self.last_state = state
        self.last_action = action

        # Execute action
        if action == 1:  # Buy
            order = self.create_order(
                asset=self.symbol,
                quantity=1,
                side="buy",
            )
            self.submit_order(order)
        elif action == 2:  # Sell
            order = self.create_order(
                asset=self.symbol,
                quantity=1,
                side="sell",
            )
            self.submit_order(order)
        # else: Hold
