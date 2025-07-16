<br>

<div align="center">

<img src="product_image.png" alt="ReinFin Logo" width="50%" height="50%">

</div>

# ReinFin
Reinforcement learning in finance.


## [0. Contents](#0._Contents) <a id='0._Contents'></a>

## [1. Introduction](#1._Introduction)

### [1.0 Reinforcemnet Learning](#1.0_Reinforcement_Learning)

## [2. Implementation](#2._Implementation)

## [3. Results](#3._Results)

## [4. Next Steps](#4._Next_Steps)

## 1. Introduction<a id='1._Introduction'></a>
The purpose of this project is to explore Reinforcement Learning in the context of finance.
More specifically, RL is commonly applied to stock trading.

### 1.0 Reinforcement Learning <a id='1.0_Reinforcement_Learning'></a>
Reinforcement Learning (RL) is a branch of Machine Learning (ML) where an agent set in an environment both observes and acts.
In contrast with other branches of ML, the RL agent takes actions that change its environment (rather than simply observing to learn a pattern/signal).
The agent learns to make decisions by interacting with an environment. 
The goal is to learn a policy that maximizes cumulative reward over time. 
At each step, the agent observes the current state, chooses an action, receives a reward, and transitions to a new state. 
Through trial and error, the agent learns which actions yield the most favorable long-term outcomes.

Some common applications of RL include 
* gaming 
* robotics
* autonomous vehicles
* finance

There is a massive amount of literature on RL, and many algorithms have been proposed and proven effective.
Currently, in this project we will implement the Dueling Double Deep Q Learning algorithm.
This is an algorithm which builds off of the Double Deep Q Network which in turn is built off of so-called Deep Q Networks.
In [Section 2.](#2._Implementation) we will discuss how we implemented this network.

### 1.1 Algorithmic Trading <a id='1.1_Algorithmic_Trading'></a>
Algorithmic trading is the use of computer programs to execute financial market trades based on predefined rules and strategies. 
These algorithms can analyze market data, identify trading opportunities, and place orders at speeds and frequencies far beyond human capabilities.
This blend of finance, data science, and software engineering is a prime area for applying machine learning and reinforcement learning to dynamically optimize decision-making.

In this project we use RL to develop a trading algorithm to make daily buy/sell/hold decisions for a single stock market symbol (SPY).
This is just one of many examples of a trading algorithm from the fast-paced domain of algorithmic trading.

## 2. Implementation <a id='2._Implementation'></a>