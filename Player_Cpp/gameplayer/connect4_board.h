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

class board {
public:
    // initialize the board
    void init();
    // if we can play a move
    bool canplay(int col);
    // update (r, c) with piece
    // @piece = -1/0/1, -1 means the maximizer, 1 means the minimizer, 0 means unoccupy
    void update(int r, int c, int piece);
    /* return the status of the board
        -1 means a win for the maximizer
        0 means a draw
        1 means a win for the minimizer
        2 means unknown 
    */
    int get_status();
    /*
        get how many 3-connected components like .XXX.
        @return (count for player -1, count for player 1)
    */
    std::pair<int, int> get_connect_3();
   /*
        get how many 3-connected components like .XXX
        @return (count for player -1, count for player 1)
   */
    std::pair<int, int> get_connect_3_weak();
   /*
        get how many 2-connected components like .XX. or ..XX or ..XX
        @return (count for player -1, count for player 1)
   */
    std::pair<int, int> get_connect_2();
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
    int status = 0;
};