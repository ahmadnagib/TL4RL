import numpy as np
import gym
import itertools
from gym import spaces
from collections import defaultdict 
import scipy.io as sio
import math
from lib import plotting
import sys

def sarsa(env, num_episodes, discount_factor, alpha, epsilon, epsilon_decay, decay_steps, loaded_qtable):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).

    # load the appropriate policy if policy reuse is applied
    if loaded_qtable == 'no':
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
    else:
        Q = loaded_qtable

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        action_probs = policy(state, epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # One step in the environment
        for t in itertools.count():
            # Print out agent's info per learning step for debugging.
            print()
            print("---Current Episode: ",i_episode)
            print()
            print("---Current Step is: ",t)
            print("---Current State is: ",state)
            print("---Q_table is: ", dict(Q))
            print("---action_probs is: ", action_probs)
            print("---Action Choice is: ", action)

            # Take a step
            next_state, reward, done, _ = env.step(action)

            print("---Reward is: ", reward)
            
            # Pick the next action
            next_action_probs = policy(next_state, epsilon)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            print()
            if done:
                break
                
            action = next_action
            state = next_state 

        # apply a simple decaying epsilon during the first decay_steps
        if i_episode < decay_steps:
            epsilon = epsilon * epsilon_decay
    return Q, stats


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation, e):

        A = np.ones(nA, dtype=float) * e / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - e)
        return A
    return policy_fn