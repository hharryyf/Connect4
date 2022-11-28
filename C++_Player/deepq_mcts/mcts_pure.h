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
#include "bit_board.h"
#include "gameplayer.h"


class mcts_pure : public gameplayer {
public:
    // this init method would let the player to initialize the board
    // turn: integer 1/-1 represents whether the player plays first or second
    void init(int turn, std::string name) {

    }
    /* 
      @previous_move: integer between 0 and max_col - 1 represents the column the opponent moves, 
                        -1 means the current move is the first move
      @return: a number between 0 and max_col - 1 represents the column of the current player is playing
    */ 
    int play(int previous_move) {
        return -1;
    }
    /*
      Must take a move at some position
      return value is equal to position
    */
    int force_play(int position) {
        return position;
    }
    /*
      @return: the name of the player
    */
    std::string display_name() {
        return this->name;
    }
    /*
      do something when the game is over, in many cases the player does nothing here
    */
    void game_over(int result) {

    }
    /* used for debug, can do nothing */
    void debug() {

    }

protected:
    std::default_random_engine rng = std::default_random_engine {};
    std::string name;
};