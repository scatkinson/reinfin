from gym import Env
import logging
from gym.spaces import Discrete, Box

import reinfin.constants as const

import numpy as np


class Environment(Env):
    def __init__(self, df, cash_at_risk):
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_steps = len(df) - 1
        self.action_space = Discrete(9)
        self.action_map = {
            0: (const.HOLD_STR, 0),
            1: (const.BUY_STR, 0.25),
            2: (const.BUY_STR, 0.5),
            3: (const.BUY_STR, 0.75),
            4: (const.BUY_STR, 1),
            5: (const.SELL_STR, 0.25),
            6: (const.SELL_STR, 0.5),
            7: (const.SELL_STR, 0.75),
            8: (const.SELL_STR, 1),
        }
        self.cash_at_risk = cash_at_risk
        self.observation_space = Box(low=0, high=1, shape=(6,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        return self._next_observation()

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array(
            [
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
                self.shares_held,
            ]
        )
        return obs / np.max(obs)  # Normalize

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["close"]

        self.resolve_action_tuple(current_price, self.action_map[action])

        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._next_observation(), reward, done, {}

    def resolve_action_tuple(self, current_price, pair):
        if pair[0] == const.BUY_STR:
            shares = (self.cash_at_risk * pair[1] * self.balance) / current_price
            self.balance -= shares * current_price
            self.shares_held += shares
        elif pair[0] == const.SELL_STR:
            shares_sold = self.shares_held * pair[1]
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
        else:
            return

    def render(self):
        logging.info(
            f"Step: {self.current_step}, Balance: {self.balance}, Shares: {self.shares_held}, Net Worth: {self.net_worth}"
        )
