#pragma once
#include <iostream>
#include <string>

/*
    configuration for the alpha-beta player
*/

namespace AlphaBetaConfig {
    // discount means how many steps we need to fill this node
    const double discount = 0.7;
    // value for 4 in a row
    const double four = 100000000.0;
    // value for 3 in a row
    const double three = 500000.0;
    // value for 2 in a row
    const double two = 80000.0;
    // value for 1 in a row
    const double one = 3000.0;
    // some "enum" values
    const int ROW = 0, COL = 1, DIAG = 2, ANTIDIAG = 3;
    // transposition table
    const int LOWER = 0, UPPER = 1, EXACT = 2, ENDGAME = 3;
    const int LOWER_ID = -11, UPPER_ID = -10, INVALID = -1;

    const std::string cache_file = "memo.txt";

    const int meaningful_depth = 5;

    const int max_cache = 1000000;
}


class ConfigObject {
public:
    
    ConfigObject &Set_alpha_beta_depth(int depth) {
        this->alpha_beta_max_depth = depth;
        return *this;
    }

    ConfigObject &Set_mcts_play_iteration(int iteration) {
        this->mcts_play_iteration = iteration;
        return *this;
    }

    ConfigObject &Set_mcts_train_iteration(int iteration) {
        this->mcts_train_iteration = iteration;
        return *this;
    }

    ConfigObject &Set_c_puct(int c) {
        this->c_puct = c;
        return *this;
    }
    
    int get_alpha_beta_max_depth() {
        return alpha_beta_max_depth;
    }

    int get_mcts_play_iteration() {
        return mcts_play_iteration;
    }

    int get_mcts_train_iteration() {
        return mcts_train_iteration;
    }

    int get_c_puct() {
        return c_puct;
    }

private:
    int alpha_beta_max_depth = 11;
    int mcts_play_iteration = 10000;
    int mcts_train_iteration = 500;
    int c_puct = 2;
};