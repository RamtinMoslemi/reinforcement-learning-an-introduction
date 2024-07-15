# Chapter 6

## [Cliff Walking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)

### figure 6.4
Plots are made using $\varepsilon = 0.1$ (for $\varepsilon$-greedy action selection), step size $\alpha=0.5$ and discount factor $\gamma = 1$. 
Each model has been trained for 500 episodes and results have been averaged over 1000 runs to be smoother.

|                                                                                                                              Sum of rewards during episodes                                                                                                                              |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/figure_6_4.png" width="583"> <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/figure_6_4+.png" width="583"> | 

#### Questions (just for fun):
- Why do the Q-learning rewards hover around -50 after convergence?
- Why do the Sarsa rewards hover around -25 after convergence?
- Why do the Q-learning rewards seem more noisy than the Sarsa rewards?


|                                                                         Optimal Path                                                                          |                                                                           Visualized                                                                           | 
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/q_learning_optimal_path.png" height="240">   |   <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/q_learning_cliff_walking.gif" height="240">   |
|     <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/sarsa_optimal_path.png" height="240">      |     <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/sarsa_cliff_walking.gif" height="240">      |
| <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/expected_sarsa_optimal_path.png" height="240"> | <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/expected_sarsa_cliff_walking.gif" height="240"> |