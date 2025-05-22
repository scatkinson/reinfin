import gym
from reinfin.agents.philbot import Agent
from reinfin.util import plot_learning_curve
from reinfin.environment.environment import Environment

import pandas as pd


if __name__ == "__main__":
    file = "data/scott-atkinson_20250521_osvC/SPY_2020-01-01_2023-12-31_Day_scott-atkinson_20250521_osvC.csv"
    df = pd.read_csv(file)
    env = Environment(df, 0.5)
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=4,
        eps_min=0.01,
        input_dims=[8],
        lr=0.003,
    )
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
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

        avg_score = np.mean(scores[-100:])

        print(
            f"episode {i} score {score},\naverage score {avg_score},\nepsilon {agent.epsilon}"
        )
    x = [i + 1 for i in range(n_games)]
    filename = "/tmp/spy_day_trade.png"
    plot_learning_curve(x, scores, filename)
