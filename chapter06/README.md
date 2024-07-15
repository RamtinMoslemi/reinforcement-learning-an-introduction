# Chapter 6

## [Cliff Walking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)

### figure 6.4
Plots are made using $\varepsilon = 0.1$ (for $\varepsilon$-greedy action selection), step size $\alpha=0.5$ and discount factor $\gamma = 1$. 
Each model has been trained for 500 episodes and results have been averaged over 1000 runs to be smoother.

| Sum of rewards during episodes|
| :---------------------------: |
| <img src="https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/figure_6_4.png" width="583" height="438"> | 

#### Questions (just for fun):
- Why do the Q-learning rewards hover around -50 after convergence?
- Why do the Sarsa rewards hover around -25 after convergence?
- Why do the Q-learning rewards seem more noisy than the Sarsa rewards?


| Q-learning | Sarsa | 
| :--------: | :---: |
| ![](https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/Optimal%20Q-Learning%20path%20for%20Cliff%20Walking.png) | ![](https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/Optimal%20Sarsa%20path%20for%20Cliff%20Walking.png) |
| ![](https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/q-learning_cliff_walking.gif) | ![](https://github.com/RamtinMoslemi/reinforcement-learning-an-introduction/blob/master/images/chapter06/sarsa_cliff_walking.gif) |

