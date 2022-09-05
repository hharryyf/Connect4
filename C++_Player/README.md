# The C++ Version of the Connect 4 Player

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project consists of a bunch of connect4 players. The project is still under development. We **plan** to implement at least the following 2 players: 

1) an alpha-beta pruning player with endgame transposition table

2) a MCTS player with Deep Learning enhancement 

We also support: human play mode in a terminal window, and a game player that allows game players of different language to compete accross a socket  
	
## Technologies
Project is created with:
* LibTorch: 1.12.1
* CMake: 3.24.1
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

At the moment, only human VS alpha-beta pruning agent is supported.