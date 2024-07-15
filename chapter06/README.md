# Chapter 6

## [Windy Gridworld](https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/environments/windy_gridworld_env.py)

### figure 6.3
Plots are made using $\varepsilon = 0.1$ (for $\varepsilon$-greedy action selection), step size $\alpha=0.5$ and discount factor $\gamma = 1$. 

| Figure 6.3 | Optimal Policy |
|:----------:| :------------: |
| ![figure_6_3](https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/figure_6_3.png) | <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/windy_gridworld_optimal_path.png" width="720"> |

| Greedy Policy Visualized | 
| :----------------------: |
| ![](https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/q_learning_windy_grid_world.gif) |

## [Cliff Walking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)

### figure 6.4
Plots are made using $\varepsilon = 0.1$ (for $\varepsilon$-greedy action selection), step size $\alpha=0.5$ and discount factor $\gamma = 1$. 
Each model has been trained for 500 episodes and results have been averaged over 1000 runs to be smoother.

| Figure 6.4 | Figure 6.4 + Expected Sarsa | 
|:----------:| :------------------------: |
| ![figure_6_4](https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/figure_6_4.png) | ![figure_6_4_+](https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/figure_6_4+.png)

#### Questions (just for fun):
- Why do the Q-learning rewards hover around -50 after convergence?
- Why do the Sarsa rewards hover around -25 after convergence?
- Why do the Q-learning rewards seem more noisy than the Sarsa rewards?


|                                                                         Optimal Path                                                                          |                                                                           Visualized                                                                           | 
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/q_learning_optimal_path.png" width="720">   |   <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/q_learning_cliff_walking.gif">   |
|     <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/sarsa_optimal_path.png" width="720">      |     <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/sarsa_cliff_walking.gif">      |
| <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/expected_sarsa_optimal_path.png" width="720"> | <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/expected_sarsa_cliff_walking.gif"> |
