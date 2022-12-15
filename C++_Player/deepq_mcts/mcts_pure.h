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
#include <memory>
#include <chrono>
#include "bit_board.h"
#include "gameplayer.h"
#include "mcts_node.h"

/*
  MCTS implemented with the C_PUCT algorithm
*/

class pure_mcts_tree {
public:
    pure_mcts_tree(int c_puct=3, int n_playout=1000) {
        this->root = std::make_shared<mcts_node>(mcts_node(nullptr, 1.0));
        this->num_playout = n_playout;
        this->c_puct = c_puct;
        this->rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
    /**
     * MCTS-Playout given the current board state
    */
    void playout(bit_board board);

    /**
     * Decide the move we are going to play given the current board state
    */
    int get_move(bit_board board);

    /**
     * Update the search tree give the current move
    */
    void update_with_move(int move);

private:

    std::vector<std::pair<int, double>> random_rollout(bit_board board) {
        auto valid_move = board.get_available();
        int len = valid_move.size();
        std::vector<std::pair<int, double>> ret(len);
        for (int i = 0 ; i < len; ++i) {
            //ret[i].second = (double) rand() / RAND_MAX;
            ret[i].second = distribution(rng);
            ret[i].first = valid_move[i];
        }

        return ret;
    }

    std::vector<std::pair<int, double>> policy_value_function(bit_board board) {
        auto vc = board.get_available();
        std::vector<std::pair<int, double>> ret;
        for (int i = 0 ; i < (int) vc.size(); ++i) {
            ret.emplace_back(vc[i], 1.0 / ((int) vc.size()));
        }

        return ret;
    }

    int evaluate_rollout(bit_board board) {
        int player = board.get_current_player();
        std::pair<bool, int> res;
        for (int i = 0 ; i < 1000; ++i) {
            res = board.has_winner();
            if (res.first) break;
            auto action_prob = random_rollout(board);
            auto best = action_prob.front();
            for (auto &p : action_prob) {
                if (p.second > best.second) best = p;
            }

            board.do_move(best.first);
        }

        if (res.second == 0) return 0;
        return res.second == board.get_current_player() ? 1 : -1;
    } 

    std::default_random_engine rng = std::default_random_engine {};
    std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(0.0, 1.0);
    std::shared_ptr<mcts_node> root;
    int num_playout;
    int c_puct;
};

class mcts_pure : public gameplayer {
public:
    // this init method would let the player to initialize the board
    // turn: integer 1/-1 represents whether the player plays first or second
    void init(int turn, std::string name, ConfigObject config) {
        this->name = name;
        this->board = bit_board();
        this->mcts = pure_mcts_tree(config.get_c_puct(), config.get_mcts_play_iteration());      
    }
    /* 
      @previous_move: integer between 0 and max_col - 1 represents the column the opponent moves, 
                        -1 means the current move is the first move
      @return: a number between 0 and max_col - 1 represents the column of the current player is playing
    */ 
    int play(int previous_move);
    /*
      @return: the name of the player
    */
    std::string display_name() {
        return this->name;
    }
    /*
      do something when the game is over, in many cases the player does nothing here
    */
    void game_over(int result);
    /* used for debug, can do nothing */
    void debug();

protected:
    std::string name;
    bit_board board;
    pure_mcts_tree mcts;
};