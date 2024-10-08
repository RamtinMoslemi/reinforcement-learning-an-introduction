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
from tqdm import trange
from utils import visualize

# probability for exploration
EPSILON = 0.1

# step size parameter
ALPHA = 0.5

# discount factor for Q-Learning and Expected Sarsa
GAMMA = 1

# environment
CLIFF_WALKING = gym.make('CliffWalking-v0')

# algorithms
ALGORITHMS = {'q_learning': 'Q-learning', 'sarsa': 'Sarsa', 'expected_sarsa': 'Expected Sarsa'}

# where to save the images
IMAGE_DIR = '../images/chapter06/'


def plot_heatmap(q_values, algorithm_name, saving_format='.png'):
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
    plt.title(f'Optimal {ALGORITHMS[algorithm_name]} path for Cliff Walking', fontsize=24)
    im = sns.heatmap(value_max, cmap="inferno", annot=labels, annot_kws={'fontsize': 26}, fmt='s')
    im.figure.savefig(f'{IMAGE_DIR}{algorithm_name}_optimal_path{saving_format}', bbox_inches='tight', dpi=200)


def choose_action(state: int, q_values: np.ndarray, env: gym.Env, epsilon: float = EPSILON) -> int:
    """Epsilon-greedy policy: selects the maximum value action with probability (1-epsilon) and selects randomly with
    epsilon probability.

    Args:
        state (integer): current state
        q_values (ndarray): current value function of shape (n_states, n_actions)
        env (Env): gymnasium environment
        epsilon (float): exploration parameter between 0 and 1

    Returns:
        action (integer): the chosen action
    """
    if np.random.random() > epsilon:  # greedy action
        action = np.argmax(q_values[state])
    else:  # random action
        action = env.action_space.sample()
    return action


def q_learning(n_episodes: int, env: gym.Env, alpha: float = ALPHA, gamma: float = GAMMA) -> (np.ndarray, np.ndarray):
    """Q-Learning: runs the Q-learning algorithm with epsilon-greedy policy on gym environment.

    Args:
        n_episodes (integer): the number of episodes we train the model
        env (Env): gym environment
        alpha (float): step size parameter between 0 and 1
        gamma (float): discount factor between 0 and 1

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
            q[state, action] += alpha * (reward + gamma * np.max(q[next_state]) - q[state, action])
            state = next_state
            episode_reward_sum += reward
            terminal = terminated or truncated
        episode_rewards[episode_i] = episode_reward_sum
    return episode_rewards, q


def sarsa(n_episodes: int, env: gym.Env, alpha: float = ALPHA, gamma: float = GAMMA) -> (np.array, np.ndarray):
    """Sarsa: runs the Sarsa algorithm with epsilon-greedy policy on gym environment.

    Args:
        n_episodes (integer): the number of episodes we train the model
        env (Env): gymnasium environment
        alpha (float): step size parameter between 0 and 1
        gamma (float): discount factor between 0 and 1

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
            q[state, action] += alpha * (reward + gamma * q[next_state, next_action] - q[state, action])
            state, action = next_state, next_action
            episode_reward_sum += reward
            terminal = terminated or truncated
        episode_rewards[episode_i] = episode_reward_sum
    return episode_rewards, q


def expected_sarsa(n_episodes: int, env: gym.Env, alpha: float = ALPHA, gamma: float = GAMMA) -> (np.array, np.ndarray):
    """Expected Sarsa: runs the Expected Sarsa algorithm with epsilon-greedy policy on gym environment.

    Args:
        n_episodes (integer): the number of episodes we train the model
        env (Env): gymnasium environment
        alpha (float): step size parameter between 0 and 1
        gamma (float): discount factor between 0 and 1

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
            q_expected = (1 - EPSILON) * np.max(q[next_state]) + EPSILON * np.mean(q[next_state])
            q[state, action] += alpha * (reward + gamma * q_expected - q[state, action])
            state = next_state
            episode_reward_sum += reward
            terminal = terminated or truncated
        episode_rewards[episode_i] = episode_reward_sum
    return episode_rewards, q


# Sarsa converges to the safe path, while Q-Learning converges to the optimal path
def figure_6_4(algorithms=None, episodes=500, runs=1000, saving_format='.png'):
    if algorithms is None:
        algorithms = [q_learning, sarsa, expected_sarsa]
    q_shape = (CLIFF_WALKING.observation_space.n, CLIFF_WALKING.action_space.n)
    rewards = {algorithm.__name__: np.zeros(episodes) for algorithm in algorithms}
    q = {algorithm.__name__: np.zeros(shape=q_shape) for algorithm in algorithms}

    # repeat for multiple runs to achieve smoother results
    for _ in trange(runs):
        for algorithm in algorithms:
            reward, q_value = algorithm(episodes, CLIFF_WALKING)
            rewards[algorithm.__name__] += reward
            q[algorithm.__name__] += q_value

    # averaging over independent runs
    for algorithm in algorithms:
        rewards[algorithm.__name__] /= runs
        q[algorithm.__name__] /= runs

    # draw reward curves
    for algorithm in algorithms:
        plt.plot(rewards[algorithm.__name__], label=ALGORITHMS[algorithm.__name__])
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig(IMAGE_DIR + 'figure_6_4+' + saving_format, bbox_inches='tight', dpi=200)
    plt.close()

    # display optimal policy
    render_env = gym.make('CliffWalking-v0', render_mode='rgb_array')
    for algorithm in algorithms:
        plot_heatmap(q[algorithm.__name__], algorithm.__name__)
        visualize.make_gif(render_env, q[algorithm.__name__], f'{IMAGE_DIR}{algorithm.__name__}_cliff_walking')


def figure_6_6(interim_runs=50_000, interim_episodes=100, asymptotic_runs=10, asymptotic_episodes=100_000,
               delta=0.05, save_format='.png'):
    step_sizes = np.arange(0.1, 1.0 + delta, delta)
    algorithms = [q_learning, sarsa, expected_sarsa]
    interim_performance = {algorithm.__name__: np.zeros(len(step_sizes)) for algorithm in algorithms}
    asymptotic_performance = {algorithm.__name__: np.zeros(len(step_sizes)) for algorithm in algorithms}

    # Interim performance
    for _ in trange(interim_runs):
        for idx, step_size in zip(range(len(step_sizes)), step_sizes):
            for algorithm in algorithms:
                reward, _ = algorithm(interim_episodes, CLIFF_WALKING, alpha=step_size)
                interim_performance[algorithm.__name__][idx] += np.mean(reward) / interim_runs

    # Asymptotic performance
    for _ in trange(asymptotic_runs):
        for idx, step_size in zip(range(len(step_sizes)), step_sizes):
            for algorithm in algorithms:
                reward, _ = algorithm(asymptotic_episodes, CLIFF_WALKING, alpha=step_size)
                asymptotic_performance[algorithm.__name__][idx] += np.mean(reward) / asymptotic_runs

    # plotting
    markers = {'q_learning': 's', 'sarsa': 'v', 'expected_sarsa': 'x'}
    colors = {'q_learning': 'black', 'sarsa': 'blue', 'expected_sarsa': 'red'}
    plt.figure(figsize=(12, 8))
    for algorithm in algorithms:
        alg = algorithm.__name__
        # Interim performance
        plt.plot(step_sizes, interim_performance[alg], marker=markers[alg], linestyle='dotted',
                 label=f'Interim Performance {ALGORITHMS[alg]}', color=colors[alg])
        # asymptotic performance
        plt.plot(step_sizes, asymptotic_performance[alg], marker=markers[alg],
                 label=f'Asymptotic Performance {ALGORITHMS[alg]}', color=colors[alg])
    plt.xlabel(r'$\alpha$')
    plt.xticks(np.arange(0.1, 1.1, 0.1))
    plt.ylabel('Sum of rewards per episode')
    plt.ylim([-150, 0])
    plt.legend()
    plt.savefig(IMAGE_DIR + 'figure_6_6_' + save_format, bbox_inches='tight', dpi=200)
    plt.close()


if __name__ == '__main__':
    figure_6_4(runs=1000)
    # Due to limited capacity of calculation of my machine, I can't complete this experiment with 100,000 episodes and
    # 50,000 runs to get the fully averaged performance, However even I only play for 10,000 episodes and 10 runs,
    # the curves looks still good.
    figure_6_6(interim_runs=1000, asymptotic_runs=10, asymptotic_episodes=10_000)
