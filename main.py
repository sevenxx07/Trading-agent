import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
 
class Memory:
    def __init__(self, cap):
        self.capacity = cap
        self.memory = []
        self.start = 0
 
    def remember(self, state, action, qvalue):
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, qvalue))
        else:
            self.memory[self.start] = (state, action, qvalue)
            self.start = (self.start+1) % self.capacity
 
    def sample(self, num):
        return random.sample(self.memory, num)
 
 
class Network(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.hidden = nn.Linear(n_observations, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, n_actions)
        self.dropout = nn.Dropout(0.35)
        self.relu = nn.ReLU()
 
    def forward(self, x):
        x = self.dropout(self.relu(self.hidden(x)))
        x = self.dropout(self.relu(self.hidden2(x)))
        x = self.dropout(self.relu(self.hidden3(x)))
        x = self.output(x)
        return x
 
 
class NeuralNet:
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.model = None
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    def build(self, obs, act):
        self.model = Network(obs, act).to(self.dev)
 
    def greedy(self, observ):
        with torch.no_grad():
            result = self.model(torch.from_numpy(observ).float().to(self.dev))
            result = result.to("cpu").detach().numpy()
            return np.argmax(result)
 
    def greedyQ(self, observ):
        with torch.no_grad():
            result = self.model(torch.from_numpy(observ).float().to(self.dev))
            result = result.to("cpu").detach().numpy()
            return np.max(result)
 
 
    def eps(self, state, epsilon, env):
        if np.random.rand() <= epsilon:
            return env.action_space.sample()
        return self.greedy(state)
 
    def batch(self, batch_size, mem, mem_size):
        if mem_size < batch_size:
            return
        mybatch = mem.sample(batch_size)
        state = torch.tensor(np.array([s[0] for s in mybatch]), device=self.dev)
        pred = self.model(state.clone().detach().float())
        actions = torch.tensor([s[1] for s in mybatch], device=self.dev).unsqueeze(1)
        pred_vals=pred.gather(1, actions).squeeze()
        target = torch.tensor([s[2] for s in mybatch], device=self.dev, dtype=torch.float32)
 
        #loss = nn.MSELoss()(pred_vals, target)
        loss = nn.SmoothL1Loss()(pred_vals, target)
        loss.backward()
        for p in self.model.parameters():
            p.grad.data.clamp_(-1, 1)
        optim = torch.optim.SGD(self.model.parameters(),lr=0.001)
        optim.step()
 
 
class TradingAgent:
    def __init__(self):
        self.actions = self.get_position_list()
        self.action_size = len(self.actions)
        self.big_constant = 800
        self.agent = None
 
    def reward_function(self, history):
        # TODO feel free to change the reward function ...
        #  This is the default one used in the gym-trading-env library, however, there might be better ones
        #  @see https://gym-trading-env.readthedocs.io/en/latest/customization.html#custom-reward-function
        return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])
 
    def compute_macd(self, close_prices, short_window=12, long_window=26, signal_window=9):
        """
        Compute the Moving Average Convergence Divergence (MACD) for a given series of close prices.
 
        Parameters:
        - close_prices (pd.Series): Series of close prices.
        - short_window (int): Short window size for MACD. Default is 12.
        - long_window (int): Long window size for MACD. Default is 26.
        - signal_window (int): Signal window size for MACD. Default is 9.
 
        Returns:
        - macd (pd.Series): Series containing MACD values.
        """
        short_ema = close_prices.ewm(span=short_window, min_periods=1).mean()
        long_ema = close_prices.ewm(span=long_window, min_periods=1).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, min_periods=1).mean()
        macd = macd_line - signal_line
        return macd
 
    def make_features(self, df):
        # TODO feel free to include your own features - for example, Bollinger bounds
        #  IMPORTANT - do not create features that look ahead in time.
        #  Doing so will result in 0 points from the homework.
        #  @see https://gym-trading-env.readthedocs.io/en/latest/features.html
 
        # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
        # this is percentual change in time
        df["feature_close"] = df["close"].pct_change()
        # Create the feature : close[t] / open[t]
        df["feature_open"] = df["close"] / df["open"]
        # Create the feature : high[t] / close[t]
        df["feature_high"] = df["high"] / df["close"]
        # Create the feature : low[t] / close[t]
        df["feature_low"] = df["low"] / df["close"]
        # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
        df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
 
        df["feature_sma"] = df["close"].rolling(window=20).mean()
        df["feature_sd"] = df["close"].rolling(window=20).std()
        df["feature_ubb"] = df["feature_sma"] + 2 * df["feature_sd"]
        df["feature_lbb"] = df["feature_sma"] - 2 * df["feature_sd"]
        #RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df["feature_rsi"] = 100 - (100 / (1 + gain / loss))
 
        df['feature_macd'] = self.compute_macd(df['close'])  # Moving Average Convergence Divergence (MACD)
 
 
        df['feature_price_spread'] = df['high'] - df['low']
        df['feature_price_momentum'] = df['close'].diff()  # Change in close price
        df['feature_ema_10'] = df['close'].ewm(span=10, adjust=False).mean()  # 10-period Exponential Moving Average
        df.dropna(inplace=True)
 
        # the library automatically adds two features - your position and
 
        return df
 
    def get_position_list(self):
        # TODO feel free to specify different set of actions
        #  here, the acceptable actions are positions -1.0, -0.9, ..., 2.0
        #  corresponding actions are integers 0, 1, ..., 30
        #  @see https://gym-trading-env.readthedocs.io/en/latest/environment_desc.html#action-space
 
        #  If position < 0: the environment performs a SHORT (by borrowing USDT and buying BTC with it).
        #  If position > 1: the environment uses MARGIN trading (by borrowing BTC and selling it to get USDT).
 
        return [x / 10.0 for x in range(-10, 21, 2)]
 
    def train(self, env, num_epochs=2, epsilon=0.02, batch_size=128, replay_mem=10000):
        # TODO implement your version of the train method
        #  you can run several epizodes ...
        #  no strict bounds set, but use Reinforcement learning techniques,
        #  the choice of algorithm is Your responsibility
        actions = env.action_space.n
        observations = env.observation_space.shape[0]
        epsilon_c = int((num_epochs*40000+1)*epsilon/(1-epsilon))
        self.agent = NeuralNet(observations, actions)
        self.agent.build(observations, actions)
        history = Memory(replay_mem)
        # Run an episode until it ends
        i = 0
        for epoch in range(num_epochs):
 
            done, truncated = False, False
            observation, info = env.reset()
            while not done and not truncated:
                # At every timestep, pick a random position index from your position, i.e., a number between -1 and 2
                #new_position = env.action_space.sample()
                #observation, reward, done, truncated, info = env.step(new_position)
                e = epsilon_c/(epsilon_c+1+i)
                action = self.agent.eps(observation, e, env)
                new_observation, reward, done, truncated, info = env.step(action)
                Qval = self.agent.greedyQ(new_observation)
                history.remember(observation, action, reward if done else reward + 0.99 * Qval)
                observation = new_observation
                self.agent.batch(batch_size, history, len(history.memory))
                i+=1
                #if len(agent.memory) > batch_size:
                    #replay
            #env.save_for_render(dir="render_logs")
 
    def get_test_position(self, observation):
        # TODO implement the method that will return position for testing
        #  In other words, this method will contain policy used for testing, not training.
        #return 20 if observation[1] > 1.05 else 10  # all in USD ... maps to position 0.0
        return self.agent.greedy(observation)
 
    def test(self, env):
        # DO NOT CHANGE - all changes will be ignored after upload to BRUTE!
        done, truncated = False, False
        observation, info = env.reset()
        while not done and not truncated:
            new_position = self.get_test_position(observation)
            observation, reward, done, truncated, info = env.step(new_position)
