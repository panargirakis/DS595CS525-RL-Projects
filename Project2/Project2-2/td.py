#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
import gym

# -------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
# -------------------------------------------------------------------------


def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #

    rand_num = random.random()

    if rand_num < epsilon:
        action = random.randrange(nA)
    else:
        action = np.argmax(Q[state])

    ############################
    return action


def sarsa(env: gym.Env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop n_episodes
    for an_episode in range(n_episodes):

        # define decaying epsilon
        epsilon *= 0.99

        # initialize the environment
        state = env.reset()

        # get an action from policy
        action = epsilon_greedy(Q, state, env.action_space.n, epsilon)

        # loop for each step of episode
        done = False
        while not done:

            # return a new state, reward and done
            new_state, reward, done, info = env.step(action)

            # get next action
            next_action = epsilon_greedy(Q, new_state, env.action_space.n, epsilon)

            # TD update, td_target, td_error, new Q
            Q[state][action] += alpha * (reward + gamma * Q[new_state][next_action] - Q[state][action])

            # update state
            state = new_state

            # update action
            action = next_action

    return Q


def q_learning(env: gym.Env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop n_episodes
    for an_episode in range(n_episodes):

        # initialize the environment
        state = env.reset()

        # loop for each step of episode
        done = False
        while not done:

            # get an action from policy
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            
            # return a new state, reward and done
            new_state, reward, done, info = env.step(action)
            
            # TD update, td_target with best Q, td_error, new Q
            Q[state][action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state][action])
            
            # update state
            state = new_state

    ############################
    return Q
