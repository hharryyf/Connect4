#pragma once 
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <stack>
#include <vector>
#include <array>
#include <set>
#include <queue>
#include <cassert>
#include <cmath>
#include <map>
#include <unordered_map>
#include <random>
#include <unordered_set>
#include <bitset>
#include <string>
// the interface for a connect4 player
class gameplayer {
    // this init method would let the player to initialize the board
    // turn: integer 1/-1 represents whether the player plays first or second
    virtual void init(int turn, std::string name) = 0;
    /* 
      @previous_move: integer between 0 and max_col - 1 represents the column the opponent moves, 
                        -1 means the current move is the first move
      @return: a number between 0 and max_col - 1 represents the column of the current player is playing
    */ 
    virtual int play(int previous_move) = 0;
    /*
      @return: the name of the player
    */
    virtual std::string display_name() = 0;
    /*
      do something when the game is over, in many cases the player does nothing here
    */
    virtual void game_over() = 0;
};