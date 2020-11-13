#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
from statistics import mean
import math
import gym
from collections import namedtuple
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN, load, save, DuelingDQN
from replay_memory import ReplayMemory

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))


class Agent_DQN(Agent):
    def __init__(self, env: gym.Env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for replay buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        super(Agent_DQN, self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.env = env
        self.n_episodes = args.n_episodes
        self.batch_size = args.batch_size
        self.init_memory = args.memory_size
        self.learning_rate = args.learning_rate
        self.memory_size = self.init_memory * 10
        self.replay_memory = ReplayMemory(self.memory_size)
        self.optimize_model_interval = 4
        self.gamma = 0.99
        self.save_interval = args.save_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rolling_reward = deque(maxlen=30)

        self.policy_net = DuelingDQN(n_actions=env.action_space.n).to(self.device)
        self.target_net = DuelingDQN(n_actions=env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.steps_done = 0
        self.target_update_int = args.target_update_int

        self.epsilon_decay = 1000000
        self.eps_end = 0.02
        self.eps_start = 1

        self.save_path = args.m_save_path

        if not os.path.isdir(os.path.dirname(self.save_path)):
            os.mkdir(os.path.dirname(self.save_path))

        self.log_save_path = args.m_save_path

        if not os.path.isdir(os.path.dirname(self.log_save_path)):
            os.mkdir(os.path.dirname(self.log_save_path))

        self.log_buffer = pd.DataFrame(columns=["Time Step", "Episode", "30-Episode Average Reward"])

        self.log_buffer.to_csv(self.log_save_path, index=False)

        if args.test_dqn or args.continue_training:
            # you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            load_path = args.m_load_path
            model = load(load_path, env.action_space.n)
            self.target_net.load_state_dict(model.state_dict())
            self.policy_net.load_state_dict(model.state_dict())
            del model

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        state = self.get_state(observation)

        if sample > eps_threshold or test:
            with torch.no_grad():
                action = self.policy_net(state.to(self.device)).max(1)[1].view(1, 1).flatten().numpy()
                try:
                    if len(action) == 1:
                        action = action[0]
                except TypeError:
                    pass
        else:
            action = random.randrange(self.env.action_space.n)

        ###########################
        return action

    def push(self, *args):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.replay_memory.push(*args)

        ###########################

    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        return self.replay_memory.sample(self.batch_size)

    @staticmethod
    def get_state(obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        for episode in range(self.n_episodes):
            obs = self.env.reset()
            total_reward = 0.0
            episode_steps = 0
            done = False
            while not done:
                episode_steps += 1
                action = self.make_action(obs, test=False)

                obs_new, reward, done, info = self.env.step(action)
                total_reward += reward
                reward = torch.tensor([reward], device=self.device)
                action_t = torch.tensor([[action]], device=self.device, dtype=torch.long)

                next_state = None
                if not done:
                    next_state = self.get_state(obs_new)
                self.push(self.get_state(obs), action_t.to(self.device),
                          next_state, reward.to(self.device))
                obs = obs_new

                if self.steps_done > self.init_memory:

                    if self.steps_done % self.optimize_model_interval == 0:
                        self.optimize_model()

                    if self.steps_done % self.target_update_int == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                    if self.steps_done % self.save_interval == 0:
                        save(self.policy_net, self.save_path)
                        print("Model saved!")

            self.rolling_reward.append(total_reward)
            mean_reward = mean(self.rolling_reward)

            if len(self.rolling_reward) == self.rolling_reward.maxlen:
                self.log_buffer = self.log_buffer.append({'Time Step': self.steps_done,
                                                          'Episode': episode,
                                                          '30-Episode Average Reward': mean_reward},
                                                         ignore_index=True)

            if episode % 20 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {} \t 30-ep avg reward: {:.3f}'.format(
                    self.steps_done, episode, episode_steps, total_reward, mean_reward))

            if episode % 50 == 0 and len(self.rolling_reward) == self.rolling_reward.maxlen:
                self.log_buffer.to_csv(self.log_save_path, mode='a', header=False, index=False)
                self.log_buffer = self.log_buffer.iloc[0:0]  # clear so that the save data does not get re-appended

        save(self.policy_net, self.save_path)
        ###########################

    def optimize_model(self):
        if len(self.replay_memory) < self.batch_size:
            raise Exception("The batch size cannot be larger than the memory size")

        transitions = self.replay_buffer()

        batch = Transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward)))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)

        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        del non_final_mask
        del non_final_next_states
        del state_batch
        del action_batch
        del reward_batch
        del state_action_values
        del next_state_values
        del expected_state_action_values
        del loss
