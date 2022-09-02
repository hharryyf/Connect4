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
#include "config.h"

class board {
public:
    // initialize the board
    void init();
    // if we can play a move
    bool canplay(int c);
    // update column c with piece
    void update(int c, int piece);
    /* return the status of the board
        -1 means a win for the maximizer
        0 means a draw
        1 means a win for the minimizer
        2 means unknown 
    */
    int get_status();
  /*
    0000000
    0000000
    0000000
    0000000
    0000000
    0000000
  */
    std::string print_board();
private:
    int a[6][7], col[7], move = 0;
};