# Connect 4
Learn Deep Reinforcement Learning by writing a Connect4 player. Although alpha-beta pruning can play a game of this size very well, 
we want to use this game as an opportunity to learn how the AlphaZero algorithm works. We would firstly learn how to code a Deep Reinforcement Learning Player in Python, because it's easier, then translate it to C++ if we have time.

## Project Outline
The project would consist of the following iterations:

1. - [x] Implement a descent C++ based alpha-beta pruning agent, we would use this agent to 

    * be familiar with a C++ project

    * act as a reference player and see how the Deep Q-Learning agent performs 

2. - [x] Implement and train the Python MCTS + Deep Reinforcement Learning agent

    * the implementation completes, however, the training is way too slow

    * the translation is a MUST do task, otherwise, we have no time to obtain a good model
    
3. - [x] Translate the Python MCTS + Deep Reinforcement Learning agent to C++, the aim of this stage include

    * [x] review the CPUCT MCTS algorithm
    
    * [x] be familiar with the bit-board representation of connect-4 (I must thank my supervisor Abdallah Saffidine for introducing this)

    * [x] be familiar with the C++ deep learning library

4. - [ ] Try to improve the performance of the C++ MCTS + Deep Reinforcement Learning agent.
    
    * We have designed a player that combines 2 players such that the first several moves are played by the DQN player while the later moves are
    played by the Alpha-Beta player 

Project status: 3 players were implemented in C++: pure-MCTS-cpuct, and alpha-beta pruning (with transposition table). The DeepQN-MCTS player
were implemented in both Python and C++.

## Connect-4 board representation speed test

For these board representations, alpha-beta also supports heuristic calculation, hence it is the most computationally expensive one

![alt text](https://github.com/hharryyf/Connect4/blob/main/images/board_time.png)

## Game results (C++ players)

**Alpha-beta vs Pure-MCTS-CPUCT (20 games)** 

Alpha-beta-depth | Pure-MCTS-CPUCT playout | Win | Loss | Draw | Winning rate
--- | --- | --- | --- | --- | ---
7 | 50,000 | 16 | 4 | 0 | 80%
7 | 200,000 | 12 | 8 | 0 | 60%
7 | 500,000 | 12 | 8 | 0 | 60%
7 | 1,000,000 | 16 | 4 | 0 | 80%
11 | 50,000 | 20 | 0 | 0 | 100%
11 | 200,000 | 17 | 3 | 0 | 85%
11 | 500,000 | 19 | 1 | 0 | 95%
11 | 1,000,000 | 17 | 3 | 0 | 85%

**Pure-MCTS-CPUCT vs Pure-MCTS-CPUCT (10 games)**

Pure-MCTS-CPUCT playout | Pure-MCTS-CPUCT playout | Win | Loss | Draw | Winning rate
--- | --- | --- | --- | --- | ---
200,000 | 50,000 | 10 | 0 | 0 | 100%
500,000| 50,000 | 7 | 2 | 1 | 75%
1,000,000| 50,000 | 9 | 1 | 0 | 90%
500,000 | 200,000 | 7 | 3 | 0 | 70%
1,000,000 | 200,000 | 7 | 3 | 0 | 70%
1,000,000 | 500,000 | 7 | 3 | 0 | 70%

**Connect-Zero vs Pure-MCTS-CPUCT (10 games)**

Training games | Connect-Zero playout | Pure-MCTS-CPUCT playout | Win | Loss | Draw | Winning rate
--- | --- | --- | --- | --- | --- | ---
100 | 1,000 | 1,000 | 4 | 5 | 1 | 45%
400 | 1,000 | 1,000 | 7 | 2 | 1 | 75%
700 | 1,000 | 1,000 | 9 | 1 | 0 | 90%
700 | 5,000 | 50,000 | 5 | 4 | 1 | 55%
700 | 5,000  | 200,000 | 1 | 9 | 0 | 10%
750 | 1,000 | 1,000 | 10 | 0 | 0 | **100%**
850 | 1,000 | 2,000 | 5 | 5 | 0 | 50%
1,000 | 1,000 | 2,000 | 7 | 2 | 1 | 75%   

**Connect-Zero (first 12 moves) + Alpha-beta (depth>=11) vs Pure-MCTS-CPUCT (10 games)**

Training games | Connect-Zero playout | Pure-MCTS-CPUCT playout | Win | Loss | Draw | Winning rate
--- | --- | --- | --- | --- | ---
700 | 5,000 | 200,000 | 7 | 2 | 1 | 75%
850 | 5,000 | 200,000 | 8 | 2 | 0 | 80%
1,000 | 5,000 | 200,000 | 10 | 0 | 0 | **100%**

Observation: The Connect-Zero player is pretty good at the starting phase of the game, but it is not good after 20 moves. I have used alpha-beta-depth-21 to monitor the connect-zero player's performance for the last 20 moves, it always makes mistakes that could change the game state from winning to losing. 

## Reference
The implementation heavily referenced https://github.com/junxiaosong/AlphaZero_Gomoku

The C++ implementation of the dirichlet distribution is copied from https://github.com/gcant/dirichlet-cpp

And I have also read the following papers to have a better understanding of the algorithm aspect of this project.

[1] Monte-Carlo Graph Search for AlphaZero https://arxiv.org/pdf/2012.11045.pdf

[2] Mastering the Game of Go without Human Knowledge https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf

[3] Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm https://arxiv.org/abs/1712.01815

[4] Score Bounded Monte-Carlo Tree Search https://cgi.cse.unsw.edu.au/~abdallahs/Papers/2010/Score%20Bounded%20Monte-Carlo%20Tree%20Search.pdf
