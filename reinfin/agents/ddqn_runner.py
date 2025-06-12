from reinfin.agents.ddqn_bot import Agent
from reinfin.util import plot_learning_curve, plot_curve
from reinfin.environment.environment import Environment
from reinfin.agents import DDQNRunnerConfig
import reinfin.constants as const

import pandas as pd
import numpy as np
import logging
import torch as T
from sklearn.preprocessing import StandardScaler


class DDQNRunner:

    def __init__(self, conf: DDQNRunnerConfig):
        self.conf = conf

    def run_ddqn(self):
        if self.conf.seed > 0:
            np.random.seed(self.conf.seed)
            logging.info(f"Setting random seed to {np.random.seed}")

        logging.info(f"Loading trade_file from {self.conf.train_file}.")
        train_df = pd.read_csv(self.conf.train_file, index_col=const.TIMESTAMP_COL)
        train_df.fillna(method="bfill", inplace=True)
        logging.info(
            f"Instantiating Environment for trade_file with cash at risk: {self.conf.cash_at_risk}."
        )
        train_env = Environment(
            train_df,
            self.conf.start_cash_balance,
            self.conf.cash_at_risk,
            self.conf.lookback,
            take_profit_threshold=self.conf.take_profit_threshold,
            stop_loss_threshold=self.conf.stop_loss_threshold,
            max_stop_loss_calls=self.conf.max_stop_loss_calls,
        )

        logging.info(f"Instantiating Agent according to config.")
        agent = Agent(
            gamma=self.conf.gamma,
            epsilon=self.conf.epsilon,
            batch_size=self.conf.batch_size,
            n_actions=len(train_env.action_map),
            eps_min=self.conf.eps_min,
            eps_dec=self.conf.eps_dec,
            input_dims=[self.conf.lookback, self.conf.n_features],
            hid_out_dims=self.conf.hid_out_dims,
            dropout=self.conf.dropout,
            lr=self.conf.lr,
            replace=self.conf.replace_cnt,
            chkpt_dir=self.conf.model_save_directory,
            pipeline_id=self.conf.pipeline_id,
        )

        if self.conf.load_checkpoint:
            agent.load_models()

        scores, eps_history, net_worths, action_history, loss_values = (
            [],
            [],
            [],
            [],
            [],
        )
        n_games = self.conf.n_games

        if n_games > 0:
            for i in range(n_games):
                logging.info(f"Running training round number {i}")
                score = 0
                episode_losses = []
                loss = np.inf
                done = False
                observation = train_env.reset()
                # reset game action counter dict
                game_action_counts = {key: 0 for key in train_env.action_map.keys()}
                while not done:
                    action = agent.choose_action(observation)
                    game_action_counts[action] += 1
                    observation_, reward, done, info = train_env.step(action)
                    score += reward
                    agent.store_transition(
                        observation, action, reward, observation_, done
                    )
                    loss = agent.learn()
                    if loss:
                        episode_losses.append(loss)
                    scores.append(score)
                    eps_history.append(agent.epsilon)
                    observation = observation_

                loss_values.append(episode_losses)
                net_worths.append(train_env.net_worth)
                action_history.append(train_env.action_memory)

                avg_score = np.mean(scores[-100:])

                logging.info(
                    f"""
                    \nEPISODE {i} 
                    \nCash Balance: {train_env.cash_balance}, 
                    \nShares Count: {train_env.shares_held}, 
                    \nDividend balance: {train_env.dividend_balance},
                    \nStop Loss Intervention Count: {train_env.stop_loss_intervention_count},
                    \nNet Worth: {train_env.net_worth},
                    \nAverage Episode Loss: {np.mean(episode_losses)}
                    """
                )
            plot_curve(scores, self.conf.train_scores_plot_path)
            plot_curve(net_worths, self.conf.train_net_worths_plot_path)
            plot_learning_curve(scores, self.conf.train_scores_learning_plot_path)
            plot_learning_curve(
                net_worths, self.conf.train_net_worths_learning_plot_path
            )
            plot_learning_curve(
                [np.mean(e_losses) for e_losses in loss_values],
                self.conf.train_loss_plot_path,
            )

            avg_actions = [
                np.mean([action_memory[i][1] for action_memory in action_history])
                for i in range(len(action_history[0]))
            ]

            plot_curve(avg_actions, self.conf.train_actions_plot_path)

        if self.conf.save_model:
            agent.save_models()

        if self.conf.eval_file:
            # evaluating agent
            eval_df = pd.read_csv(self.conf.eval_file, index_col=const.TIMESTAMP_COL)
            eval_df.fillna(method="bfill", inplace=True)
            eval_env = Environment(
                eval_df,
                self.conf.start_cash_balance,
                self.conf.cash_at_risk,
                self.conf.lookback,
                take_profit_threshold=self.conf.take_profit_threshold,
                stop_loss_threshold=self.conf.stop_loss_threshold,
                max_stop_loss_calls=self.conf.max_stop_loss_calls,
                scaler=train_env.scaler,
            )
            logging.info(f"Evaluating agent on {self.conf.eval_file}")
            score = 0
            done = False
            observation = eval_env.reset()
            rewards, scores, net_worths, action_history, loss_history = (
                [],
                [],
                [],
                [],
                [],
            )
            loss_values = np.zeros(len(eval_env.df), dtype=np.float32)
            index = 0
            # eliminate exploration for the evaluation phase
            agent.epsilon = 0
            agent.eps_min = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = eval_env.step(action)
                score += reward
                agent.store_transition(observation, action, reward, observation_, done)
                loss = agent.learn()
                loss_values[index] = loss
                observation = observation_
                rewards.append(reward)
                scores.append(score)
                net_worths.append(eval_env.net_worth)
                action_history.append(eval_env.action_map[action])

            logging.info(
                f"""
                \nEVALUATION ROUND
                \nCash Balance: {eval_env.cash_balance}, 
                \nShares Count: {eval_env.shares_held}, 
                \nDividend balance: {eval_env.dividend_balance},
                \nStop Loss Intervention Count: {eval_env.stop_loss_intervention_count},
                \nNet Worth: {eval_env.net_worth}
                \nMultiplier: {eval_env.net_worth/self.conf.start_cash_balance}
                """
            )

            logging.info(
                f"BENCHMARK Multiplier: {eval_df['close'].iloc[-1]/eval_df['close'].iloc[0]}"
            )

            plot_curve(scores, self.conf.eval_scores_plot_path)
            plot_curve(net_worths, self.conf.eval_net_worths_plot_path)
            plot_curve([x[1] for x in action_history], self.conf.eval_actions_plot_path)
