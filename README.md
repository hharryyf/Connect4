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



## Reference
The implementation heavily referenced https://github.com/junxiaosong/AlphaZero_Gomoku

And I have also read the following papers to have a better understanding of the algorithm aspect of this project.

[1] Monte-Carlo Graph Search for AlphaZero https://arxiv.org/pdf/2012.11045.pdf

[2] Mastering the Game of Go without Human Knowledge https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf

[3] Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm https://arxiv.org/abs/1712.01815

