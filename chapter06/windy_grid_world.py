#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import gymnasium as gym
from windy_gridworld_env import WindyGridworld
from matplotlib import pyplot as plt

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 1

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# where to save the images
image_dir = '../images/chapter06'


def q_learning(n_episodes: int) -> (np.ndarray, np.ndarray):
    """Q-Learning: runs the Q-Learning algorithm on the Windy Gridworld environment.

    Args:
        n_episodes (integer): the number of episodes we train the model

    Returns:
        episode_rewards (ndarray): the sum of rewards in each episode
        q (ndarray): the trained Q-values
    """
    episode_rewards = np.empty(n_episodes)

    env = WindyGridworld()
    q = np.zeros(shape=(WORLD_WIDTH * WORLD_HEIGHT, len(ACTIONS)))
    for episode_i in range(n_episodes):
        state, info = env.reset()
        episode_reward_sum, terminal = 0, False
        while not terminal:
            if np.random.random() > EPSILON:  # greedy action
                action = np.argmax(q[state])
            else:  # random action
                action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            q[state, action] += ALPHA * (reward + np.max(q[next_state]) - q[state, action])
            state = next_state
            episode_reward_sum += reward
            terminal = terminated or truncated
        episode_rewards[episode_i] = episode_reward_sum
    return episode_rewards, q


def greedy_display(q: np.ndarray) -> None:
    env = WindyGridworld(render_mode='human')
    state, info = env.reset()
    while True:
        action = np.argmax(q[state])
        state, reward, terminated, truncated, info = env.step(action)
        if truncated or terminated:
            break


def figure_6_3(saving_format='.png'):
    steps, q_value = q_learning(500)

    steps = np.add.accumulate(-steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    plt.savefig(image_dir + 'figure_6_3' + saving_format)
    plt.close()

    # show video of agent playing according to the best policy
    greedy_display(q_value)

    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            best_action = np.argmax(q_value[i * WORLD_WIDTH + j, :])
            if best_action == ACTION_UP:
                optimal_policy[-1].append('U')
            elif best_action == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif best_action == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif best_action == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))


if __name__ == '__main__':
    figure_6_3()
