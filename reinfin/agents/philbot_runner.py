from reinfin.agents.philbot import Agent
from reinfin.util import plot_learning_curve, plot_curve
from reinfin.environment.environment import Environment
from reinfin.agents import PhilbotRunnerConfig

import pandas as pd
import numpy as np
import logging


class PhilbotRunner:

    def __init__(self, conf: PhilbotRunnerConfig):
        self.conf = conf

    def run_philbot(self):
        logging.info(f"Loading trade_file from {self.conf.trade_file}.")
        file = self.conf.trade_file
        df = pd.read_csv(file)
        logging.info(
            f"Instantiating Environment for trade_file with cash at risk: {self.conf.cash_at_risk}."
        )
        env = Environment(df, self.conf.cash_at_risk)
        logging.info(f"Instantiating Agent according to config.")
        agent = Agent(
            gamma=self.conf.gamma,
            epsilon=self.conf.epsilon,
            batch_size=self.conf.batch_size,
            n_actions=len(env.action_map),
            eps_min=self.conf.eps_min,
            eps_dec=self.conf.eps_dec,
            input_dims=self.conf.input_dims,
            lr=self.conf.lr,
        )
        scores, eps_history, net_worths, action_history = [], [], [], []
        n_games = self.conf.n_games

        for i in range(n_games):
            logging.info(f"Running trading game number {i}")
            score = 0
            done = False
            observation = env.reset()
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_

            scores.append(score)
            eps_history.append(agent.epsilon)
            net_worths.append(env.net_worth)
            action_history.append(env.action_memory)

            avg_score = np.mean(scores[-100:])

            logging.info(
                f"episode {i} score {score},\naverage score {avg_score},\nepsilon {agent.epsilon},\nnet worth: {env.net_worth}"
            )
        plot_curve(scores, self.conf.scores_plot_path)
        plot_curve(net_worths, self.conf.net_worths_plot_path)

        avg_actions = [
            np.mean([action_memory[i][1] for action_memory in action_history])
            for i in range(len(action_history[0]))
        ]

        plot_curve(avg_actions, self.conf.actions_plot_path)
