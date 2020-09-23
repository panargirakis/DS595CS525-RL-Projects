### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np
from typing import Dict
import sys

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS: int, nA: int, policy: np.array, gamma=0.9, tol=1e-8):
    """Evaluate the value function from action given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(state) - prev_value_function(state)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[state] is
        the value of state state
    """

    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #

    # the max difference over the state space of current iteration (will decrease over time)
    max_diff = sys.float_info.max

    # train while the difference is higher than the threshold
    while max_diff > tol:
        # new value function
        new_val_func = np.zeros(nS)

        # diff for the current session (will be increasing during each iteration)
        diff = 0

        # loop over state space
        for state in range(nS):

            # To accumulate bellman expectation
            val_acc = np.sum(policy[state] * __action_values(P, state, nA, value_function, gamma))

            # !!! ALTERNATE IMPLEMENTATION - DOES NOT USE __action_values() HELPER FUNCTION
            # val_acc = 0
            # # loop over possible actions
            # for action in range(nA):
            #     # get transitions [(prob, next_state, reward, done)]
            #     transitions = P[state][action]
            #
            #     # loop over possible outcomes of an action (scholastic transition)
            #     for prob, next_state, reward, _ in transitions:
            #         # apply bellman expectation equation
            #         val_acc += policy[state][action] * prob * (reward + gamma * value_function[next_state])

            # get the biggest difference over state space
            diff = max(diff, abs(val_acc - value_function[state]))

            # update state-value
            new_val_func[state] = val_acc

        # the new value function
        value_function = new_val_func

        # save the max diff
        max_diff = diff



    ############################
    return value_function


def __action_values(P, state, nA, value_func, gamma):
    action_values = np.zeros(nA)
    # find the value of each function
    for action in range(nA):
        transitions = P[state][action]
        # to get the value, take into account the probability
        for prob, next_state, reward, _ in transitions:
            action_values[action] += prob * (reward + gamma * value_func[next_state])

    return action_values


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.zeros([nS, nA])
    ############################
    # YOUR IMPLEMENTATION HERE #

    action_per_state = []

    for state in range(nS):
        # get the value of each action
        action_values = __action_values(P, state, nA, value_from_policy, gamma)

        # now determine the best action to take
        action_per_state.append(np.argmax(action_values))

    # convert action indices to one-hot vector encoding
    action_per_state = np.array(action_per_state)
    new_policy[np.arange(action_per_state.size), action_per_state] = 1.0

    ############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #

    is_stable = False
    V = None

    while not is_stable:
        V = policy_evaluation(P, nS, nA, new_policy, gamma, tol)
        temp_policy = policy_improvement(P, nS, nA, V, gamma)

        is_stable = np.allclose(new_policy, temp_policy)
        new_policy = temp_policy

    ############################
    return new_policy, V


def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """

    V_new = V.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #

    max_diff = sys.float_info.max

    while max_diff > tol:
        diff = 0
        v_temp = V_new.copy()
        for state in range(nS):
            action_values = __action_values(P, state, nA, V_new, gamma)
            new_val = np.max(action_values)

            diff = max(diff, abs(new_val - V_new[state]))

            v_temp[state] = new_val

        max_diff = diff
        V_new = v_temp

    policy_new = policy_improvement(P, nS, nA, V_new, gamma)

    ############################
    return policy_new, V_new


def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset() # initialize the episode
        done = False
        while not done:
            if render:
                env.render() # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #

    return total_rewards



