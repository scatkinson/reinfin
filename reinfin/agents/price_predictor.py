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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
                if col
                not in [const.CLOSE_COL, const.TRADE_COUNT_COL, const.PRED_CLOSE_COL]
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
                if col
                not in [const.TRADE_COUNT_COL, const.CLOSE_COL, const.PRED_CLOSE_COL]
            ]
        ]
        self.Y_eval = self.eval_df[const.CLOSE_COL]

        self.y_train_pred = np.zeros(len(self.Y_train))
        self.y_eval_pred = np.zeros(len(self.Y_eval))

        self.loss_values = []

    def run_price_predictor(self):
        self.train_model()
        self.eval_model()
        self.save_model()

    def train_model(self):
        pass

    def eval_model(self):
        logging.info("Obtaining in-sample forecast predictions")
        self.y_train_pred = self.model.predict_in_sample(X=self.X_train).to_numpy()

        self.y_eval_pred = self.model.predict(
            len(self.X_eval), X=self.X_eval
        ).to_numpy()
        eval_r2 = r2_score(self.y_eval_pred, self.Y_eval)
        eval_mape = loss_MAPE(self.y_eval_pred, self.Y_eval.to_numpy())
        logging.info(f"EVAL R^2: {eval_r2}")
        logging.info(f"EVAL MAPE: {eval_mape}")

        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(
            pd.to_datetime(self.Y_train.index), self.Y_train.to_numpy(), label="True"
        )
        ax.plot(pd.to_datetime(self.Y_train.index), self.y_train_pred, label="Pred")

        fig.suptitle("True v Predicted Train Stock Prices")
        ax.set_xlabel("Time")
        ax.set_ylabel("Stock price in USD")

        # Set major ticks every 10 days
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=40))

        # Format the major tick labels (e.g., as 'YYYY-MM-DD')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        logging.info(
            f"Saving train figure at {self.conf.train_forecast_plot_filename}."
        )
        fig.savefig(self.conf.train_forecast_plot_path)

        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(pd.to_datetime(self.Y_eval.index), self.Y_eval.to_numpy(), label="True")
        ax.plot(pd.to_datetime(self.Y_eval.index), self.y_eval_pred, label="Pred")

        fig.suptitle("True v Predicted Eval Stock Prices")
        ax.set_xlabel("Time")
        ax.set_ylabel("Stock price in USD")

        # Set major ticks every 10 days
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=40))

        # Format the major tick labels (e.g., as 'YYYY-MM-DD')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        logging.info(f"Saving eval figure at {self.conf.eval_forecast_plot_filename}.")
        fig.savefig(self.conf.eval_forecast_plot_path)

    def save_model(self):
        train_out_df = pd.DataFrame(
            self.y_train_pred, index=self.Y_train.index, columns=[const.PRED_CLOSE_COL]
        )
        logging.info(
            f"Saving predicted in-sample prices to {self.conf.train_predictions_path}."
        )
        train_out_df.to_csv(self.conf.train_predictions_path)
        eval_out_df = pd.DataFrame(
            self.y_eval_pred, index=self.Y_eval.index, columns=[const.PRED_CLOSE_COL]
        )
        logging.info(
            f"Saving predicted eval prices to {self.conf.eval_predictions_path}."
        )
        eval_out_df.to_csv(self.conf.eval_predictions_path)
