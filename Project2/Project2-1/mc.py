#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import gym
import random
from collections import defaultdict

# -------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implement an AI player for Blackjack.
    The main goal of this problem is to get familiar with Monte-Carlo algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v mc_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
# -------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hits otherwise
    
    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation

    curr_sum, dealer_face_up_card, has_usable_ace = observation

    action = 1
    if curr_sum >= 20:
        action = 0

    ############################
    return action 


def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function 
        by using Monte Carlo first visit algorithm.
    
    Parameters:
    -----------
    policy: function
        A function that maps an observation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for an_episodo in range(n_episodes):

        # initialize the episode
        state = env.reset()


        # generate empty episode list
        states_actions_rewards = []

        # loop until episode generation is done
        done = False
        while not done:
            # select an action
            action = policy(state)

            # return a reward and new state
            new_state, reward, done, info = env.step(action)

            # append state, action, reward to episode
            states_actions_rewards.append((state, action, reward))

            # update state to new state
            state = new_state

        # loop for each step of episode, t = T-1, T-2,...,0
        G = 0
        states_visited = set()
        for state, _, reward in reversed(states_actions_rewards):
            
            # compute G
            G = G * gamma + reward

            # unless state_t appears in states
            if state not in states_visited:
            
                # update return_count
                returns_count[state] += 1
                
                # update return_sum
                returns_sum[state] += G

                # calculate average return for this state over all sampled episodes
                V[state] = returns_sum[state] / returns_count[state]

                states_visited.add(state)



    ############################
    
    return V


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
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
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


def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts. 
        Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.    
    """
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #

    # define decaying epsilon

    for an_episode in range(1, n_episodes + 1):
        epsilon_decay = epsilon - (0.1 / an_episode)

        # initialize the episode
        state = env.reset()

        # generate empty episode list
        state_action_reward = []

        # loop until one episode generation is done
        done = False
        while not done:
            # get an action from epsilon greedy policy
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon_decay)

            # return a reward and new state
            new_state, reward, done, info = env.step(action)

            # append state, action, reward to episode
            state_action_reward.append((state, action, reward))

            # update state to new state
            state = new_state



        G = 0
        states_visited = set()
        # loop for each step of episode, t = T-1, T-2, ...,0
        for state, action, reward in reversed(state_action_reward):

            # compute G
            G = gamma * G + reward

            # unless the pair state_t, action_t appears in <state action> pair list
            if (state, action) not in states_visited:

                # update return_count
                returns_count[(state, action)] += 1

                # update return_sum
                returns_sum[(state, action)] += G

                # calculate average return for this state over all sampled episodes
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

                states_visited.add(state)
        
    return Q
