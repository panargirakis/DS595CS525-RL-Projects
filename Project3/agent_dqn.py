#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from itertools import count
import os
import sys
import math
import gym
from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model2 import DQN, DQNbn
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
        self.gamma = args.gamma
        self.render = args.render
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNbn(n_actions=env.action_space.n).to(self.device)
        self.target_net = DQNbn(n_actions=env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.steps_done = 0
        self.target_update_int = args.target_update_int

        self.epsilon_decay = 1000000
        self.eps_end = 0.02
        self.eps_start = 1

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #

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

        if sample > eps_threshold:
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
            for t in count():
                action = self.make_action(obs)

                if self.render:
                    self.env.render()

                obs_n, reward, done, info = self.env.step(action)

                total_reward += reward

                reward = torch.tensor([reward], device=self.device)

                action_t = torch.tensor([[action]], device=self.device, dtype=torch.long)

                next_state = None
                if not done:
                    next_state = self.get_state(obs_n)
                self.push(self.get_state(obs), action_t.to(self.device),
                          next_state, reward.to(self.device))
                obs = obs_n

                if self.steps_done > self.init_memory:
                    self.optimize_model()

                    if self.steps_done % self.target_update_int == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                if done:
                    break
            if episode % 20 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.steps_done, episode, t,
                                                                                     total_reward))

        ###########################

    def optimize_model(self):
        if len(self.replay_memory) < self.batch_size:
            raise Exception("The batch size cannot be larger than the memory size")
        transitions = self.replay_buffer()
        """
        zip(*transitions) unzips the transitions into
        Transition(*) creates new named tuple
        batch.state - tuple of all the states (each state is a tensor)
        batch.next_state - tuple of all the next states (each state is a tensor)
        batch.reward - tuple of all the rewards (each reward is a float)
        batch.action - tuple of all the actions (each action is an int)    
        """
        batch = Transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward)))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.uint8)

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
