from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from alpaca_trade_api import REST
from timedelta import Timedelta
from reinfin.finbert_utils import estimate_sentiment

from datetime import datetime

API_KEY = "*****"
API_SECRET = "*****"
BASE_URL = "https://paper-api.alpaca.markets/v2"

SYMBOL = "SPY"
CASH_AT_RISK = 0.75
AVARICE = 0.20
RISK_TOLERANCE = 0.05

ALPACA_CREDS = {"API_KEY": API_KEY, "API_SECRET": API_SECRET, "PAPER": True}


class MLTrader(Strategy):

    def initialize(self, symbol: str = SYMBOL, cash_at_risk: float = CASH_AT_RISK):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime("%Y-%m-%d"), three_days_prior.strftime("%Y-%m-%d")

    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == "positive" and probability > 0.999:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    # order_type="bracket",
                    take_profit_price=last_price * (1 + AVARICE),
                    stop_loss_price=last_price * (1 - RISK_TOLERANCE),
                )
                self.submit_order(order)
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > 0.999:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    # order_type="bracket",
                    take_profit_price=last_price * (1 - AVARICE),
                    stop_loss_price=last_price * (1 + RISK_TOLERANCE),
                )
                self.submit_order(order)
                self.last_trade = "sell"


start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

broker = Alpaca(ALPACA_CREDS)

strategy = MLTrader(
    name="mlstrat",
    broker=broker,
    parameters={"symbol": SYMBOL, "cash_at_risk": CASH_AT_RISK},
)

strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": SYMBOL, "cash_at_risk": CASH_AT_RISK},
)

# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
