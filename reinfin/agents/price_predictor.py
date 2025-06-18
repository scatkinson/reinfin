import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
from pmdarima import auto_arima

import logging

from reinfin.agents.price_predictor_config import PricePredictorConfig
import reinfin.constants as const
from reinfin.util import plot_curve, plot_learning_curve


def loss_MAPE(output, target):
    # MAPE loss
    return np.mean(np.abs((target - output) / target))


class PricePredictor:
    def __init__(self, conf: PricePredictorConfig):
        self.conf = conf
        self.train_df = pd.read_csv(self.conf.train_file, index_col=const.TIMESTAMP_COL)
        self.eval_df = pd.read_csv(self.conf.eval_file, index_col=const.TIMESTAMP_COL)

        self.X_train = self.train_df[
            [
                col
                for col in self.train_df.columns
                if col not in [const.CLOSE_COL, const.TRADE_COUNT_COL]
            ]
        ]
        self.Y_train = self.train_df[const.CLOSE_COL]

        self.model = auto_arima(
            self.Y_train,
            X=self.X_train,
            start_p=self.conf.start_p,
            d=self.conf.d,
            start_q=self.conf.start_q,
            max_p=self.conf.max_p,
            max_d=self.conf.max_d,
            max_q=self.conf.max_q,
            start_P=self.conf.start_P,
            D=self.conf.D,
            start_Q=self.conf.start_Q,
            max_P=self.conf.max_P,
            max_D=self.conf.max_D,
            max_Q=self.conf.max_Q,
            m=self.conf.m,
            seasonal=self.conf.seasonal,
            error_action=self.conf.error_action,
            trace=self.conf.trace,
            suppress_warnings=self.conf.suppress_warnings,
            stepwise=self.conf.stepwise,
            random_state=self.conf.seed,
            n_fits=self.conf.n_fits,
        )

        self.X_eval = self.eval_df[
            [
                col
                for col in self.eval_df.columns
                if col not in [const.TRADE_COUNT_COL, const.CLOSE_COL]
            ]
        ]
        self.Y_eval = self.eval_df[const.CLOSE_COL]

        self.y_pred = np.zeros(len(self.Y_eval))

        self.loss_values = []

    def run_price_predictor(self):
        self.train_model()
        self.eval_model()
        self.save_model()

    def train_model(self):
        pass

    def eval_model(self):
        self.y_pred = self.model.predict(len(self.X_eval), X=self.X_eval).to_numpy()
        eval_r2 = r2_score(self.y_pred, self.Y_eval)
        eval_mape = loss_MAPE(self.y_pred, self.Y_eval.to_numpy())
        logging.info(f"EVAL R^2: {eval_r2}")
        logging.info(f"EVAL MAPE: {eval_mape}")

    def save_model(self):
        out_df = pd.DataFrame(self.y_pred, index=self.Y_eval.index)
        logging.info(f"Saving predicted prices to {self.conf.model_path}.")
        out_df.to_csv(self.conf.model_path)
