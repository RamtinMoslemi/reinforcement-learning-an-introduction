#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import seaborn as sns
import gymnasium as gym
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import visualize

# probability for exploration
EPSILON = 0.1

# step size parameter
ALPHA = 0.5

# discount factor for Q-Learning and Expected Sarsa
GAMMA = 1

# environment
CLIFF_WALKING = gym.make('CliffWalking-v0')

# where to save the images
IMAGE_DIR = '../images/chapter06/'


def plot_heatmap(q_values, title, saving_format='.png'):
    # Generate heatmap showing maximum value at each state
    dim_x, dim_y = 12, 4
    action_max = q_values.argmax(axis=1)
    value_max = q_values.max(axis=1).reshape(dim_y, dim_x)
    # act_dict = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
    act_dict = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    labels = np.array([act_dict.get(action, '') for action in action_max])
    labels[37:-1], labels[-1] = ' ', 'G'
    labels = labels.reshape(dim_y, dim_x)
    plt.figure(figsize=(18, 6))
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.axis('equal')
    plt.title(title, fontsize=24)
    im = sns.heatmap(value_max, cmap="inferno", annot=labels, annot_kws={'fontsize': 26}, fmt='s')
    im.figure.savefig(IMAGE_DIR + title + saving_format, bbox_inches='tight', dpi=200)


def choose_action(state: int, q_values: np.ndarray, env: gym.Env) -> int:
    """Epsilon-greedy policy: selects the maximum value action with probability (1-epsilon) and selects randomly with
    epsilon probability.

    Args:
        state (integer): current state
        q_values (ndarray): current value function of shape (n_states, n_actions)
        env (Env): gymnasium environment

    Returns:
        action (integer): the chosen action
    """
    if np.random.random() > EPSILON:  # greedy action
        action = np.argmax(q_values[state])
    else:  # random action
        action = env.action_space.sample()
    return action


def q_learning(n_episodes: int, env: gym.Env) -> (np.ndarray, np.ndarray):
    """Q-Learning: runs the Q-learning algorithm with epsilon-greedy policy on gym environment.

    Args:
        n_episodes (integer): the number of episodes we train the model
        env (Env): gym environment

    Returns:
        episode_rewards (ndarray): the sum of rewards in each episode
        q (ndarray): the trained Q-values
    """
    episode_rewards = np.empty(n_episodes)

    q = np.zeros(shape=(env.observation_space.n, env.action_space.n))

    for episode_i in range(n_episodes):
        state, info = env.reset()
        episode_reward_sum, terminal = 0, False
        while not terminal:
            action = choose_action(state, q, env)
            next_state, reward, terminated, truncated, info = env.step(action)
            q[state, action] += ALPHA * (reward + GAMMA * np.max(q[next_state]) - q[state, action])
            state = next_state
            episode_reward_sum += reward
            terminal = terminated or truncated
        episode_rewards[episode_i] = episode_reward_sum
    return episode_rewards, q


def sarsa(n_episodes: int, env: gym.Env) -> (np.array, np.ndarray):
    """Sarsa: runs the Sarsa algorithm with epsilon-greedy policy on gym environment.

    Args:
        n_episodes (integer): the number of episodes we train the model
        env (Env): gymnasium environment

    Returns:
        episode_rewards (ndarray): the sum of rewards in each episode
        q (ndarray): the trained Q-values
    """
    episode_rewards = np.empty(n_episodes)

    q = np.zeros(shape=(env.observation_space.n, env.action_space.n))

    for episode_i in range(n_episodes):
        state, info = env.reset()
        episode_reward_sum, terminal = 0, False
        action = choose_action(state, q, env)
        while not terminal:
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = choose_action(next_state, q, env)
            q[state, action] += ALPHA * (reward + GAMMA * q[next_state, next_action] - q[state, action])
            state, action = next_state, next_action
            episode_reward_sum += reward
            terminal = terminated or truncated
        episode_rewards[episode_i] = episode_reward_sum
    return episode_rewards, q


# Sarsa converges to the safe path, while Q-Learning converges to the optimal path
def figure_6_4(episodes=500, runs=1000, saving_format='.png'):
    q_shape = (CLIFF_WALKING.observation_space.n, CLIFF_WALKING.action_space.n)
    sarsa_rewards, sarsa_q = np.zeros(episodes), np.zeros(shape=q_shape)
    q_learning_rewards, q_learning_q = np.zeros(episodes), np.zeros(shape=q_shape)

    # repeat for multiple runs to achieve smoother results
    for _ in tqdm(range(runs)):
        sarsa_rewards_, sarsa_q_ = sarsa(episodes, CLIFF_WALKING)
        sarsa_rewards += sarsa_rewards_
        sarsa_q += sarsa_q_
        q_learning_rewards_, q_learning_q_ = q_learning(episodes, CLIFF_WALKING)
        q_learning_rewards += q_learning_rewards_
        q_learning_q += q_learning_q_

    # averaging over independent runs
    sarsa_rewards /= runs
    sarsa_q /= runs
    q_learning_rewards /= runs
    q_learning_q /= runs

    # draw reward curves
    plt.plot(sarsa_rewards, label='Sarsa', color='deepskyblue')
    plt.plot(q_learning_rewards, label='Q-learning', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig(IMAGE_DIR + 'figure_6_4' + saving_format, bbox_inches='tight', dpi=200)
    plt.close()

    # display optimal policy
    render_env = gym.make('CliffWalking-v0', render_mode='rgb_array')
    plot_heatmap(sarsa_q, f'Optimal Sarsa path for Cliff Walking')
    visualize.make_gif(render_env, sarsa_q, 'sarsa_cliff_walking', image_dir=IMAGE_DIR)
    plot_heatmap(q_learning_q, f'Optimal Q-learning path for Cliff Walking')
    visualize.make_gif(render_env, q_learning_q, 'q-learning_cliff_walking', image_dir=IMAGE_DIR)


# Due to limited capacity of calculation of my machine, I can't complete this experiment
# with 100,000 episodes and 50,000 runs to get the fully averaged performance
# However even I only play for 1,000 episodes and 10 runs, the curves looks still good.
def figure_6_6():
    step_sizes = np.arange(0.1, 1.1, 0.1)
    episodes = 1000
    runs = 10

    ASY_SARSA = 0
    ASY_EXPECTED_SARSA = 1
    ASY_QLEARNING = 2
    INT_SARSA = 3
    INT_EXPECTED_SARSA = 4
    INT_QLEARNING = 5
    methods = range(0, 6)

    performance = np.zeros((6, len(step_sizes)))
    for run in range(runs):
        for ind, step_size in tqdm(list(zip(range(0, len(step_sizes)), step_sizes))):
            q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
            q_expected_sarsa = np.copy(q_sarsa)
            q_q_learning = np.copy(q_sarsa)
            for ep in range(episodes):
                sarsa_reward = sarsa(q_sarsa, expected=False, step_size=step_size)
                expected_sarsa_reward = sarsa(q_expected_sarsa, expected=True, step_size=step_size)
                q_learning_reward = q_learning(q_q_learning, step_size=step_size)
                performance[ASY_SARSA, ind] += sarsa_reward
                performance[ASY_EXPECTED_SARSA, ind] += expected_sarsa_reward
                performance[ASY_QLEARNING, ind] += q_learning_reward

                if ep < 100:
                    performance[INT_SARSA, ind] += sarsa_reward
                    performance[INT_EXPECTED_SARSA, ind] += expected_sarsa_reward
                    performance[INT_QLEARNING, ind] += q_learning_reward

    performance[:3, :] /= episodes * runs
    performance[3:, :] /= 100 * runs
    labels = ['Asymptotic Sarsa', 'Asymptotic Expected Sarsa', 'Asymptotic Q-Learning',
              'Interim Sarsa', 'Interim Expected Sarsa', 'Interim Q-Learning']

    for method, label in zip(methods, labels):
        plt.plot(step_sizes, performance[method, :], label=label)
    plt.xlabel('alpha')
    plt.ylabel('reward per episode')
    plt.legend()

    plt.savefig('../images/figure_6_6.png')
    plt.close()


if __name__ == '__main__':
    figure_6_4()