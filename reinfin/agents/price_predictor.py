import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score

import logging

from reinfin.agents.price_predictor_config import PricePredictorConfig
import reinfin.constants as const
from reinfin.util import plot_curve, plot_learning_curve


def loss_MAPE(output, target):
    # MAPE loss
    return T.mean(T.abs((target - output) / target))


class LSTMModel(nn.Module):
    def __init__(
        self, input_size, hidden_sizes, num_layers, output_size, lr, dropout_size_list
    ):
        super(LSTMModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.lstm = nn.LSTM(input_size, hidden_sizes[0], num_layers, batch_first=True)

        self.hidden_layers_list = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i + 1 == len(hidden_sizes):
                self.hidden_layers_list.append(
                    nn.Linear(in_features=hidden_sizes[i], out_features=output_size)
                )
            else:
                self.hidden_layers_list.append(
                    nn.Linear(
                        in_features=hidden_sizes[i], out_features=hidden_sizes[i + 1]
                    )
                )

        self.dropout_list = nn.ModuleList()
        for dropout_size in dropout_size_list:
            self.dropout_list.append(nn.Dropout(dropout_size))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = loss_MAPE
        logging.info(f"Is CUDA available? {T.cuda.is_available()}")
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        for fc, dropout in zip(self.hidden_layers_list, self.dropout_list):
            lstm_out = F.relu(fc(lstm_out))
            lstm_out = dropout(lstm_out)
        return lstm_out


class PricePredictor:
    def __init__(self, conf: PricePredictorConfig):
        self.conf = conf
        self.train_df = pd.read_csv(self.conf.train_file, index_col=const.TIMESTAMP_COL)
        self.eval_df = pd.read_csv(self.conf.eval_file, index_col=const.TIMESTAMP_COL)

        self.model = LSTMModel(
            input_size=len(self.train_df.columns) - 2,
            hidden_sizes=self.conf.hidden_sizes,
            num_layers=self.conf.num_layers,
            output_size=1,
            lr=self.conf.learning_rate,
            dropout_size_list=self.conf.dropout_list,
        )

        self.X_train = T.tensor(
            self.train_df[
                [
                    col
                    for col in self.train_df.columns
                    if col not in [const.TRADE_COUNT_COL, const.CLOSE_COL]
                ]
            ].values,
            dtype=T.float32,
        ).to(self.model.device)
        self.Y_train = T.tensor(
            self.train_df[const.CLOSE_COL].values, dtype=T.float32
        ).to(self.model.device)

        self.dataset_train = TensorDataset(self.X_train, self.Y_train)

        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.conf.batch_size,
        )

        self.X_eval = T.tensor(
            self.eval_df[
                [
                    col
                    for col in self.eval_df.columns
                    if col not in [const.TRADE_COUNT_COL, const.CLOSE_COL]
                ]
            ].values,
            dtype=T.float32,
        ).to(self.model.device)
        self.Y_eval = T.tensor(
            self.eval_df[const.CLOSE_COL].values, dtype=T.float32
        ).to(self.model.device)

        self.dataset_eval = TensorDataset(self.X_eval, self.Y_eval)

        self.dataloader_eval = DataLoader(
            self.dataset_eval,
            batch_size=self.conf.batch_size,
        )

        self.loss_values = []

    def run_price_predictor(self):
        self.train_model()
        self.eval_model()
        self.save_model()

    def train_model(self):
        logging.info("Training Model")
        self.model.train()

        for epoch in range(self.conf.epochs):
            epoch_losses = []
            for i, (inputs, labels) in enumerate(self.dataloader_train):
                outputs = self.model(inputs)
                loss = self.model.loss(outputs, labels)
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                self.loss_values.append(loss.item())
                epoch_losses.append(loss.item())
            if (epoch + 1) % 200 == 0:
                logging.info(f"Completed EPOCH {epoch + 1}")
                logging.info(
                    f"EPOCH {epoch + 1} average loss: {np.mean(epoch_losses)}."
                )
        logging.info(f"Saving train loss curve at {self.conf.train_loss_plot_path}.")
        plot_learning_curve(self.loss_values, self.conf.train_loss_plot_path)

    def eval_model(self):
        eval_loss = []
        with T.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloader_eval):
                outputs = self.model(inputs)
                loss = self.model.loss(outputs, labels)
                eval_loss.append(loss.item())
        logging.info(f"Average eval loss: {np.mean(eval_loss)}.")
        y_pred = self.model(self.X_eval)
        y_pred_numpy = y_pred.detach().numpy()
        eval_error = root_mean_squared_error(y_pred_numpy, self.Y_eval.detach().numpy())
        eval_r2 = r2_score(y_pred_numpy, self.Y_eval.detach().numpy())
        eval_mape = loss_MAPE(y_pred, self.Y_eval).item()
        logging.info(f"EVAL RMSE: {eval_error}.")
        logging.info(f"EVAL R^2: {eval_r2}.")
        logging.info(f"EVAL MAPE: {eval_mape}.")

    def save_model(self):
        logging.info(f"Saving model at {self.conf.model_path}")
        T.save(self.model.state_dict(), self.conf.model_path)
