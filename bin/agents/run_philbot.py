import gym
from reinfin.agents.philbot import Agent
from reinfin.util import plot_learning_curve

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
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
    filename = "/tmp/lunar_lander.png"
    plot_learning_curve(x, scores, filename)
