from gym import Env
import logging
from gym.spaces import Discrete, Box

import reinfin.constants as const

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Environment(Env):
    def __init__(
        self,
        df,
        start_cash_balance,
        cash_at_risk,
        lookback,
        take_profit_threshold=np.inf,
        scaler=None,
    ):
        self.df = df.reset_index(drop=True)
        if scaler:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(self.df)
        self.current_step = 0
        self.start_cash_balance = start_cash_balance
        self.cash_balance = start_cash_balance
        self.shares_held = 0
        self.dividend_balance = 0
        self.net_worth = self.cash_balance
        self.last_net_worth = self.cash_balance
        self.max_steps = len(df) - 1
        self.action_space = Discrete(9)
        self.action_map = {
            0: (const.HOLD_STR, 0),
            1: (const.BUY_STR, 0.25),
            2: (const.BUY_STR, 0.5),
            3: (const.BUY_STR, 0.75),
            4: (const.BUY_STR, 1),
            5: (const.SELL_STR, -0.25),
            6: (const.SELL_STR, -0.5),
            7: (const.SELL_STR, -0.75),
            8: (const.SELL_STR, -1),
        }
        self.cash_at_risk = cash_at_risk
        self.lookback = lookback
        self.take_profit_threshold = take_profit_threshold
        self.observation_space = Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.action_memory = []
        self.shares_held_memory = np.zeros(len(self.df), dtype=np.float32)
        self.cash_balance_memory = np.zeros(len(self.df), dtype=np.float32)
        self.dividend_memory = np.zeros(len(self.df), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.cash_balance = self.start_cash_balance
        self.shares_held = 0
        self.dividend_balance = 0
        self.net_worth = self.cash_balance
        self.action_memory = []
        return self._next_observation()

    def _next_observation(self):
        if self.current_step >= self.lookback:
            rows = pd.DataFrame(
                self.scaler.transform(
                    self.df.iloc[
                        self.current_step + 1 - self.lookback : self.current_step + 1
                    ]
                ),
                columns=self.df.columns,
            )
            shares_held = self.shares_held_memory[
                self.current_step + 1 - self.lookback : self.current_step + 1
            ]
            balance = self.cash_balance_memory[
                self.current_step + 1 - self.lookback : self.current_step + 1
            ]
        else:
            rows = pd.DataFrame(
                self.scaler.transform(self.df.iloc[: self.current_step + 1]),
                columns=self.df.columns,
            )
            row_pad = pd.DataFrame(
                self.scaler.transform(self.df.iloc[[self.current_step]]),
                columns=self.df.columns,
            )
            row_pad_list = [row_pad] * (self.lookback - (self.current_step + 1))
            rows = pd.concat([rows] + row_pad_list, axis=0)

            shares_held = self.shares_held_memory[: self.current_step + 1]
            shares_held = np.pad(
                shares_held,
                (0, self.lookback - (self.current_step + 1)),
                constant_values=shares_held[self.current_step],
            )
            balance = self.cash_balance_memory[: self.current_step + 1]
            balance = np.pad(
                balance,
                (0, self.lookback - (self.current_step + 1)),
                constant_values=balance[self.current_step],
            )
        arr_list = [balance, shares_held] + [
            rows[col].to_numpy() for col in self.df.columns if col != "trade_count"
        ]
        obs = np.array(arr_list)
        obs = obs.transpose().flatten()
        # return obs / np.max(obs)  # Normalize
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["close"]

        self.resolve_action_tuple(current_price, self.action_map[action])
        self.shares_held_memory[self.current_step] = self.shares_held
        self.cash_balance_memory[self.current_step] = self.cash_balance

        self.check_take_profit(current_price)
        self.dividend_memory[self.current_step] = self.dividend_balance

        self.net_worth = (
            self.cash_balance + self.shares_held * current_price + self.dividend_balance
        )
        reward = (self.net_worth - self.last_net_worth) / self.last_net_worth

        self.current_step += 1
        done = self.current_step >= self.max_steps

        if self.current_step % 500 == 0:
            self.render(current_price)
            logging.info(f"Last reward: {reward}.")
            logging.info(f"Last net worth: {self.last_net_worth}.")
        self.last_net_worth = self.net_worth

        self.action_memory.append(self.action_map[action])

        return self._next_observation(), reward, done, {""}

    def resolve_action_tuple(self, current_price, pair):
        if pair[0] == const.BUY_STR:
            # no buys if not enough cash
            if self.cash_balance < 1:
                return
            shares = (self.cash_at_risk * pair[1] * self.cash_balance) / current_price
            # if shares * current_price >= const.LOGGING_THRESHOLD:
            #     self.render()
            #     logging.info(
            #         f"Purchasing {shares} shares at current price {current_price} for {shares * current_price} dollars."
            #     )
            self.cash_balance -= shares * current_price
            self.shares_held += shares
        elif pair[0] == const.SELL_STR:
            shares_sold = self.shares_held * (-1) * pair[1]
            self.cash_balance += shares_sold * current_price
            self.shares_held -= shares_sold
            # if shares_sold * current_price >= const.LOGGING_THRESHOLD:
            #     self.render()
            #     logging.info(
            #         f"Selling {shares_sold} at current price {current_price} for {shares_sold*current_price} dollars."
            #     )
        else:
            return

    def render(self, current_price):
        logging.info(
            f"\nStep: {self.current_step}, \nCash Balance: {self.cash_balance}, \nShares: {self.shares_held}, \nCurrent Price: {current_price}, \nDividend Balance: {self.dividend_balance}, \nNet Worth: {self.net_worth}"
        )

    def check_take_profit(self, current_price):
        # sell off shares if portfolio hits a specified profit and deposit in a dividends account
        if (
            self.shares_held * current_price + self.cash_balance
            > (1 + self.take_profit_threshold) * self.start_cash_balance
            and self.shares_held * current_price
            > self.take_profit_threshold * self.start_cash_balance
        ):
            logging.info(
                f"Portfolio value exceeded {1 + self.take_profit_threshold}x growth"
            )
            shares_to_sell = (
                self.start_cash_balance * self.take_profit_threshold / current_price
            )
            logging.info(
                f"Selling {shares_to_sell} shares at price {current_price} and depositing profit {shares_to_sell * current_price} into dividend account"
            )
            self.shares_held -= shares_to_sell
            self.dividend_balance += (
                self.start_cash_balance * self.take_profit_threshold
            )
            self.render(current_price)
