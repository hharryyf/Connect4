# The C++ Version of the Connect 4 Player

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project consists of a bunch of connect4 players. The project is still under development. We **plan** to implement at least the following 3 players: 

1) an alpha-beta pruning player with endgame transposition table

2) a pure MCTS player

3) a MCTS player with Deep Learning enhancement 

We also support: human play mode in a terminal window

Optional task: A game player that allows game players of different language to compete accross a socket  
	
## Technologies
Project is created with:
* LibTorch: >= 1.12.1
* CMake: >= 3.24.1
* C++ 14
	
## Setup
To run this project, download LibTorch in any directory you preferred. After that assume you are in the Connect4/C++_Player directory, do the following.

```
$ mkdir build
$ cmake -DCMAKE_PREFIX_PATH=/the absolute path of /libtorch/ ..
$ cmake --build . --config Release
$ cd Release
$ ./main
```

At the moment, 2 players were implemented in C++: pure-MCTS-cpuct, and alpha-beta pruning (with transposition table) agent and the DeepQN-MCTS player
were implemented in python.

Game results:

alpha-beta-depth-11 vs Pure-MCTS 

Alpha-beta-depth | Pure-MCTS playout | Win | Loss | Draw | Winning rate
--- | --- | --- | --- | --- | ---
11 | 50,000 | 10 | 0 | 0 | 100%
11 | 200,000 | 8 | 2 | 0 | 80%
11 | 500,000 | 9 | 1 | 0 | 90%
11 | 1,000,000 | 0 | 0 | 0 | N/A

Pure-MCTS vs Pure-MCTS

Pure-MCTS playout | Pure-MCTS playout | Win | Loss | Draw | Winning rate
--- | --- | --- | --- | --- | ---
200,000 | 50,000 | 0 | 0 | 0 | 0.0
500,000| 50,000 | 0 | 0 | 0 | 0.0
1,000,000| 50,000 | 0 | 0 | 0 | 0.0
500,0000 | 50,000 | 0 | 0 | 0 | 0.0
500,0000 | 200,000 | 0 | 0 | 0 | 0.0
1,000,0000 | 500,000 | 0 | 0 | 0 | 0.0





