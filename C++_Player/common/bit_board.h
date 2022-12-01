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

/**
 * The implementation of bit-board
 * Supports: make a move, judge the winner if it exists
 * Does not support: undo a move
 * 
 * The board representation
 * Note that 7, 15, 23, 31, 39, 47, 55 are empty
 * 
 * 0  1  2  3  4  5  6  7
 * 8  9 10 11 12 13 14 15
 *16 17 18 19 20 21 22 23
 *24 25 26 27 28 29 30 31  
 *32 33 34 35 36 37 38 39 
 *40 41 42 43 44 45 46 47 
 *
*/

class bit_board {
public:
    bit_board() {
        for (int i = 0 ; i < 7; ++i) this->col[i] = -1;
        this->current_player = 1;
        this->status = 2;
        this->last_move = -1;
        this->x_board = this->o_board = 0;
        this->num_move = 0;
    }

    /**
     * check if we can make a move at a column
    */
    bool can_move(int move) {
        return (move < 7) && (move >= 0) && (this->col[move] < 5);
    }

    /**
     * return the list of columns that are available
    */
    std::vector<int> get_available() {
        std::vector<int> ret;
        for (int i = 0 ; i < 7; ++i) {
            if (col[i] < 5) ret.push_back(i);
        }

        return ret;
    }

    /**
     * check if we have a winner (if there's a winner?, game result)
    */
    std::pair<bool, int> has_winner() {
        if (this->status != 2) return std::make_pair(true, this->status);
        return std::make_pair(false, 2);
    }

    /**
     * check if the game ends
    */
    bool game_end() {
        return this->status != 2;
    }

    /**
     * the column of the last move
    */
    int get_last_move() {
        return this->last_move;
    }

    
    /**
     * take a move in column "move"
     * 0 <= move <= 6, and -1 <= col[move] < 6
     * 
     * update the status of the board after this move
    */
    bool do_move(int move) {
        assert(!game_end());
        if (!can_move(move)) return false;  
        this->col[move]++;
        this->last_move = move;
        this->num_move++;
        if (this->current_player == 1) {
            // if it is the turn of X-player, do some or operation on the x_board
            this->x_board |= (1ull << (this->col[move] * 8 + move));
        } else {    
            // if it is the turn of O-player, do some or operation on the o_board
            this->o_board |= (1ull << (this->col[move] * 8 + move));
        }

        if (x_win()) {
            this->status = 1;
        } else if (o_win()) {
            this->status = -1;
        } else if (this->num_move == 42) {
            this->status = 0;
        }

        this->current_player *= -1;
        return true;
    }

    bit_board duplicate() {
        bit_board new_board = bit_board();
        new_board.x_board = this->x_board;
        new_board.o_board = this->o_board;
        new_board.current_player = this->current_player;
        new_board.status = this->status;
        new_board.num_move = this->num_move;
        for (int i = 0 ; i < 7; ++i) new_board.col[i] = this->col[i];
        return new_board;
    }

    /*
        Get the total number of moves
    */
    int get_move() {
        return this->num_move;
    }

    /*
        Get the player making the next move
    */
    int get_current_player() {
        return this->current_player;
    }

    void debug() {
        printf("column information: ");
        for (int i = 0 ; i < 7; ++i) printf("%d ", col[i]);
        printf("\ncurrent player = %d, status = %d, last_move = %d, num_move = %d\n", current_player, status, last_move, num_move);
    }

protected:
    int col[7], current_player, status, last_move, num_move;
    unsigned long long x_board, o_board;
    
    bool win_row(unsigned long long player_board) {
        return (player_board) & (player_board << 1) & (player_board << 2) & (player_board << 3);
    }

    bool win_col(unsigned long long player_board) {
        return (player_board) & (player_board << 8) & (player_board << 16) & (player_board << 24);
    }

    bool win_diag(unsigned long long player_board) {
        return (player_board) & (player_board << 7) & (player_board << 14) & (player_board << 21);
    }

    bool win_antidiag(unsigned long long player_board) {
        return (player_board) & (player_board << 9) & (player_board << 18) & (player_board << 27);
    }

    bool x_win() {
        return win_row(this->x_board) || win_col(this->x_board) || win_diag(this->x_board) || win_antidiag(this->x_board);
    }

    bool o_win() {
        return win_row(this->o_board) || win_col(this->o_board) || win_diag(this->o_board) || win_antidiag(this->o_board);
    }
};