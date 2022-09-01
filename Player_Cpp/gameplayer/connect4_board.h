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
    bool canplay(int col);
    // update col with piece
    void update(int col, int piece);
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
    void print_bord();
private:
    int a[max_row][max_col], col[max_col];
};