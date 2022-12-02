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
    
3. - [ ] Translate the Python MCTS + Deep Reinforcement Learning agent to C++, the aim of this stage include

    * review the CPUCT MCTS algorithm
    
    * be familiar with the bit-board representation of connect-4 (I must thank my supervisor Abdallah Saffidine for introducing this)

    * be familiar with the C++ deep learning library

Project status: 2 players were implemented in C++: pure-MCTS-cpuct, and alpha-beta pruning (with transposition table). The DeepQN-MCTS player
were implemented in python.

## Game results

**Alpha-beta vs Pure-MCTS-CPUCT (10 games)** 

Alpha-beta-depth | Pure-MCTS-CPUCT playout | Win | Loss | Draw | Winning rate
--- | --- | --- | --- | --- | ---
11 | 50,000 | 10 | 0 | 0 | 100%
11 | 200,000 | 8 | 2 | 0 | 80%
11 | 500,000 | 9 | 1 | 0 | 90%
11 | 1,000,000 | 7 | 3 | 0 | 70%

**Pure-MCTS-CPUCT vs Pure-MCTS-CPUCT (10 games)**

Pure-MCTS-CPUCT playout | Pure-MCTS-CPUCT playout | Win | Loss | Draw | Winning rate
--- | --- | --- | --- | --- | ---
200,000 | 50,000 | 10 | 0 | 0 | 100%
500,000| 50,000 | 7 | 2 | 1 | 75%
1,000,000| 50,000 | 9 | 1 | 0 | 90%
500,0000 | 200,000 | 7 | 3 | 0 | 70%
1,000,0000 | 200,000 | 7 | 3 | 0 | 70%
1,000,0000 | 500,000 | 7 | 3 | 0 | 70%



## Reference
The implementation heavily referenced https://github.com/junxiaosong/AlphaZero_Gomoku

And I have also read the following papers to have a better understanding of the algorithm aspect of this project.

[1] Monte-Carlo Graph Search for AlphaZero https://arxiv.org/pdf/2012.11045.pdf

[2] Mastering the Game of Go without Human Knowledge https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf

[3] Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm https://arxiv.org/abs/1712.01815

