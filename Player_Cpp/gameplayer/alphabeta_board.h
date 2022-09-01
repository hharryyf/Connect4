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

class alphabeta_board {
public:

    void init() {

    }

    void update(int r, int c, int piece) {

    }

    int status() {
        return -1;
    }

private:

    struct gameboard {
        int a[max_row][max_col], height[max_col], move = 0;
        int diag[max_row][max_col], antidiag[max_row][max_col];
        void init() {
            int i, j;
            for (i = 0 ; i < max_row; ++i) {
                for (j = 0 ; j < max_col; ++j) {
                    a[i][j] = 0;
                }
            }
        }

        bool canplay(int column) {
            return height[column] < max_row - 1 && column >= 0 && column < max_col;
        }

        void play(int column, int piece) {
            assert(canplay(column));
            a[column][++height[column]] = piece;
            ++move;
        }

        void undomove(int column) {
            assert(height[column] >= 0);
            a[column][height[column]] = 0;
            --height[column];
            --move;
        }
    };

    struct group {
        std::vector<std::pair<int, int>> window[max_row + max_col - 1];
        int status[max_row + max_col - 1];
        double score[max_row + max_col - 1];
        int x_win = 0, o_win = 0;
        double tol_score = 0;
        void init() {

        }

        void recalculate(int index, gameboard &brd) {
            
        }

        int get_status() {
            return -1;
        }
    };

    group diag, antidiag, row, col;
    gameboard board;
};