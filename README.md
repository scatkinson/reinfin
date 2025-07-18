<br>

<div align="center">

<img src="product_image.png" alt="ReinFin Logo" width="50%" height="50%">

</div>

# ReinFin
Reinforcement learning in finance.


## [0. Contents](#0._Contents) <a id='0._Contents'></a>

## [1. Introduction](#1._Introduction)

* ### [1.0 Reinforcemnet Learning](#1.0_Reinforcement_Learning)
* ### [1.1 Algorithmic Trading](#1.1_Algorithmic_Trading)

## [2. Code Base](#2._Code_Base)

* ### [2.0 `bin`](#2.0_bin)
  * #### [2.0.0 `conf`](#2.0.0_conf)
* ### [2.1 `data`](#2.1_data)
* ### [2.2 `images`](#2.2_images)
* ### [2.3 `logs`](#2.3_logs)
* ### [2.4 `model`](#2.4_model)
* ### [2.5 `reinfin`](#2.5_reinfin)
  * #### [2.5.0 `agents`](#2.5.0_agents)
  * #### [2.5.1 `environment`](#2.5.1_environment)
  * #### [2.5.2 `extract`](#2.5.2_extract)
  * #### [2.5.3 `processing`](#2.5.3_processing)
  * #### [2.5.4 Top-level utilities](#2.5.4_Top-level_utilities)

## [3. DDQN Discussion](#3._DDQN_discussion)

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

Key concepts:
* **Agent:** The decision-maker.
* **Environment:** The scenario with which the agent interacts.
* **Reward:** Feedback signal for each action.
* **Policy:** The strategy that the agent follows to select actions.

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

## 2. Code Base <a id='2._Code_Base'></a>

### 2.0 `bin` <a id='2.0_bin'></a>

Each subdirectory in the `bin` directory contains the end-user Python scripts to be executed along with the `conf` subdirectory corresponding config files.
These subdirectories are
* `agents`: contains the end-user scripts for running the DDQN agent and the Price Predictor agent.
* `extract`: contains the end-user script for running the data extract.
* `processing`: contains the end-user script for obtaining the technical indicators as additional features for the data. 

#### 2.0.0 `conf` <a id='2.0.0_conf'></a>

The `conf` subdirectory contains the `.yml` config file to be passed via the `-c` flag to the appropriate `.py` end-user script.
Here is an example of how an end-user script is executed on the command line:

`python bin/agents/ddqn_runner.py -c bin/agents/conf/ddqn_test_config.yml`

### 2.1 `data` <a id='2.1_data'></a>

The `data` directory contains the stock data files. This is where the `extract` script delivers the extracted data,
and it is where the `processing` scripts deliver the processed data files.

### 2.2 `images` <a id='2.2_images'></a>

The `images` directory contains the image files generated from scripts like the `ddqn_runn.py` and `price_predictor.py` scripts. 

### 2.3 `logs` <a id='2.3_logs'></a>

The `logs` directory is where all script logs are delivered. 
Every time a script is executed, a logfile is generated and saved in an appropriate subdirectory of the `logs` directory.
To avoid collisions each logfile's file name is appended with a `pipeline_id` that is randomly generated (unless it is set using the script's config file).

### 2.4 `model` <a id='2.4_model'></a>

The model directory is where the trained `ddqn` models are saved (given the appropriate config option is properly set).

### 2.5 'reinfin' <a id='2.5_reinfin'></a>

The `reinfin` directory is a Python library housing all the scripts for the project.

#### 2.5.0 `agents` <a id='2.5.0_agents'></a>
Holds trading agent implementations:

* `ddqn_bot.py`: The main Dueling Double DQN agent logic.

* `ddqn_runner.py`: Script to train, evaluate, and visualize a DDQN agent.

* `price_predictor.py`: A supervised learning model for price prediction baseline. Utilizes Auto-ARIMA.

* `tradingbot.py`: This is a simplified trading bot that uses the `alpaca_trade_api` and `lumibot` packages. 
Serves as a secondary benchmark for DDQN agent performance.

Accompanied by separate config modules (`*_config.py`).

#### 2.5.1 `environment` <a id='2.5.1_environment'></a>
Contains a Gym-like custom trading environment to simulate market interactions for the RL agent.

#### 2.5.2 `extract` <a id='2.5.2_extract'></a>
Extracts historical stock price data from external sources and stores it in the `data` directory.
The extract also includes sentiment scores of relevant news headlines for the given stock symbol.

* `extractor.py`: Core extraction logic.

* `extractor_config.py`: Configuration for symbol, date range, etc.

#### 2.5.3 `processing` <a id='2.5.3_processing'></a>
Responsible for data transformation and feature engineering--mainly financial technical indicators.

* `tech_indicators.py`: Appends financial indicators as features.

* `tech_indicators_config.py`: Controls the indicators to compute and how.

#### 2.5.4 Top-level utilities <a id='2.5.4_Top-level_utilities'></a>

* `config.py` / `configamend.py`: Load and dynamically modify YAML configuration files.

* `constants.py`: Global constants like default paths, column names, etc.

* `finbert_utils.py`: Helper functions to integrate sentiment from FinBERT.

* `log_wu.py`: Logging utilities.

* `util.py`: Miscellaneous utilities shared across modules.

## 3. DDQN Discussion <a id='3._DDQN_discussion'></a>

Let $\mathcal{S}$ denote the state space of the game--all possible positions a player could find themselves in (e.g., various stock metrics leading up to trade time).