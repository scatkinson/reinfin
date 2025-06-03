from reinfin.agents.ddqn_bot import Agent
from reinfin.util import plot_learning_curve, plot_curve
from reinfin.environment.environment import Environment
from reinfin.agents import DDQNRunnerConfig

import pandas as pd
import numpy as np
import logging
import torch as T


class DDQNRunner:

    def __init__(self, conf: DDQNRunnerConfig):
        self.conf = conf

    def run_ddqn(self):
        if self.conf.seed > 0:
            np.random.seed(self.conf.seed)

        logging.info(f"Loading trade_file from {self.conf.train_file}.")
        file = self.conf.train_file
        df = pd.read_csv(file)
        logging.info(
            f"Instantiating Environment for trade_file with cash at risk: {self.conf.cash_at_risk}."
        )
        train_env = Environment(
            df, self.conf.start_balance, self.conf.cash_at_risk, self.conf.lookback
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

        scores, eps_history, net_worths, action_history = [], [], [], []
        n_games = self.conf.n_games

        if n_games > 0:
            for i in range(n_games):
                logging.info(f"Running training round number {i}")
                score = 0
                done = False
                observation = train_env.reset()
                while not done:
                    action = agent.choose_action(observation)
                    observation_, reward, done, info = train_env.step(action)
                    score += reward
                    agent.store_transition(
                        observation, action, reward, observation_, done
                    )
                    agent.learn()
                    observation = observation_

                scores.append(score)
                eps_history.append(agent.epsilon)
                net_worths.append(train_env.net_worth)
                action_history.append(train_env.action_memory)

                avg_score = np.mean(scores[-100:])

                logging.info(
                    f"\nEPISODE {i} score {score},\naverage score {avg_score},\nepsilon {agent.epsilon},\nnet worth: {train_env.net_worth}"
                )
            plot_curve(scores, self.conf.train_scores_plot_path)
            plot_curve(net_worths, self.conf.train_net_worths_plot_path)
            plot_learning_curve(scores, self.conf.train_scores_learning_plot_path)
            plot_learning_curve(
                net_worths, self.conf.train_net_worths_learning_plot_path
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
            eval_df = pd.read_csv(self.conf.eval_file)
            eval_env = Environment(
                eval_df,
                self.conf.start_balance,
                self.conf.cash_at_risk,
                self.conf.lookback,
            )
            logging.info(f"Evaluating agent on {self.conf.eval_file}")
            score = 0
            done = False
            observation = eval_env.reset()
            scores, net_worths, action_history = [], [], []
            # eliminate exploration for the evaluation phase
            agent.epsilon = 0
            agent.eps_min = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = eval_env.step(action)
                score += reward
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_
                scores.append(score)
                net_worths.append(eval_env.net_worth)
                action_history.append(eval_env.action_map[action])

            logging.info(
                f"BENCHMARK Multiplier: {list(eval_df['close'])[-1]/eval_df['close'][0]}"
            )

            logging.info(f"EVAL Final Net Worth: {eval_env.net_worth}.")
            logging.info(
                f"EVAL Multiplier: {eval_env.net_worth/self.conf.start_balance}."
            )

            plot_curve(scores, self.conf.eval_scores_plot_path)
            plot_curve(net_worths, self.conf.eval_net_worths_plot_path)
            plot_curve([x[1] for x in action_history], self.conf.eval_actions_plot_path)
