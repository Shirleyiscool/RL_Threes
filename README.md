# Reinforcement Learning in "Threes"

## Presentation

[![Presentation](https://img.youtube.com/vi/W3iLLTlRbUA/0.jpg)](https://www.youtube.com/watch?v=W3iLLTlRbUA)

<hr>

### What is "Threes" and how it plays?

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/7/74/Threes_video_game_trailer.gif">
</p>

"Threes", as the prototype of a well-known game 2048, is a mobile game in a 4 by 4 grid environment. 
Similar to 2048, "Threes" require the player to swipe up/down/left/right to move the tiles. 
There are numbers on the tiles and two tiles will be merged if the numbers on them the same. 
The goal of these games is to reach as high score as you can, 
which is determined by the number value of the tiles on the grid.
But different from 2048, a swipe in "Threes" will not move the tiles to the end of the grid, 
but by one grid. Also, all the numbers are the exponent of 2 in 2048, while the numbers of "Threes" 
start from 1, 2, 3, 6, 12, ... , ![3*2^n](https://latex.codecogs.com/svg.latex?3*2^n)

### Problem Statement
 - Can we make a computer agent to learn how to play the game "Threes"?
 - If it can learn the game, can it play the game better than humans?

The goal for this project is to solve the above problem with Reinforcement Learning methods.

### Reinforcement Framework

 - Environment: A 4X4 grid (a 4X4 matrix in the model)
 - Agent: Humans or Computer
 - State: the flatten matrix with number (0 means no tile)
 - Action: Up/Down/Left/Right (WSAD control)
 - Reward: The changed (score + maximum number) after each move
 - Final Score: Sum of 3^Log(2(n(i)/3)+1), (n(i): the number in the grid that is larger than 2.)

### Project Results and Analysis

<p align="center">
<img src="https://github.com/Shirleyiscool/RL_Threes/blob/master/img/model_results.jpg?raw=true">
</p>

 - Metrics: Mean score and max score for a great number of games
 - Baseline: Random mean score: around 300; Random max score: around 3000
 - Greedy Mode: The agent selects the action that can maximize the score of the next state. 
 In general, greedy mode is slightly better than the baseline.
 - Q-learning: The agent can barely learn and got the lowest mean score, even worse than the baseline.
 **Analysis:** Pure q-learning can not work for "Threes", which includes infinite (continuous) state spaces.
 It is so hard for the agent to meet the same state again and thus hard to update the best move for each state.
 - DQN: It can make the agent learn the game and reaches around 850+ mean score and 48,000+ max score after
 50 epochs. DQN helps approximate Q(s, a) and thus the agent can learn through neural network.
 - Human level: After only 10 games (Of course I don't have so much energy to play thousands of game like
 the computer), I got 5500+ mean score and 20,000+ max score.
 
 ### Conclusions
 - DQN is better than Q-learning, but both cannot beat humans. 
 - While training DQN, it may need the whole night to train the model. So saving models would be a good habit when playing with deep learning.
 - Comprehensive dynamic programming may be also a good way to solve “Threes”, given the performance of greedy mode.
 - “Threes” is a game with infinite (continuous) states and finite (Discrete) actions, which is not a trivial task for q-learning. The dimensionality of state space is too high to use local approximators. Fitted q iteration may be competitive algorithms for this kind of problem.

<hr>

 ### Codes
 - [`agent.py`](codes/agent.py) contains a class for agent.
 - [`env.py`](codes/env.py) contains the "Threes" environment.
 - [`train_ql.py`](codes/train_ql.py) contains the functions that train the q-learning model.
 - [`dqn.py`](codes/dqn.py) contains the functions that train the Deep Q-Network algorithms
 - [`main.py`](codes/main.py) print the results of "Random"/"Greedy"/"Q-learning"/"DQN" modes
 
Since training q-learning and DQN models take time, [`train_dqn_colab.ipynb`](notebooks/train_dqn_colab.ipynb) 
and [`train_ql_colab.ipynb`](notebooks/train_dqn_colab.ipynb) are two notebooks that put in the Colab to run.
[`run_demo.ipynb`](notebooks/run_demo.ipynb) can be used to try the demo game, including
interactive mode(human), random mode, and greedy mode.

### Future Directions
- Improve the Environment: Let the agent know the next generated number like the real game "Threes" does.
- Improve DQN and try different hidden layers and activation functions.
- Try comprehensive Dynamic Programming methods.
  
  
 
 
 




