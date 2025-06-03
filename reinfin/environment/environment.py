from gym import Env
import logging
from gym.spaces import Discrete, Box

import reinfin.constants as const

import numpy as np
import pandas as pd


class Environment(Env):
    def __init__(self, df, start_balance, cash_at_risk, lookback):
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.start_balance = start_balance
        self.balance = start_balance
        self.shares_held = 0
        self.net_worth = self.balance
        self.last_net_worth = self.balance
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
        self.observation_space = Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.action_memory = []
        self.shares_held_memory = np.zeros(len(self.df), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.start_balance
        self.shares_held = 0
        self.net_worth = self.balance
        self.action_memory = []
        return self._next_observation()

    def _next_observation(self):
        if self.current_step >= self.lookback:
            rows = self.df.iloc[
                self.current_step + 1 - self.lookback : self.current_step + 1
            ]
            shares_held = self.shares_held_memory[
                self.current_step + 1 - self.lookback : self.current_step + 1
            ]
        else:
            rows = self.df.iloc[: self.current_step + 1]
            row_pad = self.df.iloc[[self.current_step]]
            row_pad_list = [row_pad] * (self.lookback - (self.current_step + 1))
            rows = pd.concat([rows] + row_pad_list, axis=0)

            shares_held = self.shares_held_memory[: self.current_step + 1]
            shares_held = np.pad(
                shares_held,
                (0, self.lookback - (self.current_step + 1)),
                constant_values=shares_held[self.current_step],
            )
        obs = np.array(
            [
                rows["open"].to_numpy(),
                rows["high"].to_numpy(),
                rows["low"].to_numpy(),
                rows["close"].to_numpy(),
                rows["volume"].to_numpy(),
                shares_held,
            ]
        )
        obs = obs.transpose().flatten()
        # return obs / np.max(obs)  # Normalize
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["close"]

        self.resolve_action_tuple(current_price, self.action_map[action])
        self.shares_held_memory[self.current_step] = self.shares_held

        self.net_worth = self.balance + self.shares_held * current_price
        reward = (self.net_worth - self.last_net_worth) / self.last_net_worth

        self.current_step += 1
        done = self.current_step >= self.max_steps

        if self.current_step % 500 == 0:
            self.render()
            logging.info(f"Last reward: {reward}.")
            logging.info(f"Last net worth: {self.last_net_worth}.")
        self.last_net_worth = self.net_worth

        self.action_memory.append(self.action_map[action])

        return self._next_observation(), reward, done, {""}

    def resolve_action_tuple(self, current_price, pair):
        if pair[0] == const.BUY_STR:
            # no buys if not enough cash
            if self.balance < 1:
                return
            shares = (self.cash_at_risk * pair[1] * self.balance) / current_price
            # if shares * current_price >= const.LOGGING_THRESHOLD:
            #     self.render()
            #     logging.info(
            #         f"Purchasing {shares} shares at current price {current_price} for {shares * current_price} dollars."
            #     )
            self.balance -= shares * current_price
            self.shares_held += shares
        elif pair[0] == const.SELL_STR:
            shares_sold = self.shares_held * (-1) * pair[1]
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            # if shares_sold * current_price >= const.LOGGING_THRESHOLD:
            #     self.render()
            #     logging.info(
            #         f"Selling {shares_sold} at current price {current_price} for {shares_sold*current_price} dollars."
            #     )
        else:
            return

    def render(self):
        logging.info(
            f"Step: {self.current_step}, Balance: {self.balance}, Shares: {self.shares_held}, Net Worth: {self.net_worth}"
        )
