#pragma once
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <tuple>
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
#include "connect4_board.h"
#include "bit_board.h"
#include "gameplayer.h"
#include "mcts_node.h"
#include "dirichlet.h"
#pragma warning(push, 0)
#include <torch/torch.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/script.h>
#pragma warning(pop)

class policy_value_net {
public:
    policy_value_net(std::string filename, std::string optname, double lr=0.002, double decay=0.0001) {
        this->module = torch::jit::load(filename);
        printf("load the jit model from path: %s\n", filename.c_str());
        for (const auto& params : module.parameters()) {
	        this->parameters.push_back(params);
        }
        
        this->optimizer = std::make_shared<torch::optim::Adam>(this->parameters, torch::optim::AdamOptions(lr).weight_decay(decay));
        if (optname != "") {
             torch::load(*this->optimizer, optname);
        }
    }
    
    /**
     * Evaluate a board position, used before the mcts expansion
     * @board: bit_board, the current bit_board state
     * @return: a tuple <vector of <move, move probablity> pair, winning probablity of the current player at this position>
    */
    std::tuple<std::vector<std::pair<int, double>>, double> evaluate_position(bit_board &board);

    /**
     * Evaluate a batch, return the move probablity of a batch and also the position score of a batch
     * @batch: a vector of 3 * 6 * 7 vector
     * @return: a tuple <vector of size batch size * 7, vector of batch size * 1>
    */

    std::tuple<std::vector<std::vector<double>>, std::vector<double>> evaluate_batches(std::vector<std::vector<std::vector<std::vector<double>>>> &batch);


    /**
     * @batch: a batch of board position vector
     * @mcts_probablity: a batch of mcts_probablity
     * @winner: a batch of reward {-1/0/1}
     * @return: a tuple used for monitoring <loss, entropy>
    */
    std::tuple<double, double> train_step(std::vector<std::vector<std::vector<std::vector<double>>>> &batch, 
                                         std::vector<std::vector<double>> &mcts_probability, 
                                         std::vector<int> &winner);

    void save_model(std::string filename, std::string optname);

    void set_train();

    void set_eval();

private:

    torch::jit::Module module;
    std::vector<at::Tensor> parameters;
    std::shared_ptr<torch::optim::Optimizer> optimizer;
};

class mcts_zero_tree {
public:
    mcts_zero_tree(int c_puct=3, int n_playout=1000) {
        this->num_playout = n_playout;
        this->c_puct = c_puct;
        this->root = std::make_shared<mcts_node>(mcts_node(nullptr, 1.0));
    }

    void attach_policy_value_function(std::shared_ptr<policy_value_net> &net) {
        this->network = net;
    }
    
    void playout(bit_board board, bool call_minimax);
    
    std::tuple<std::vector<int>, std::vector<double>> get_move_probability(bit_board board, double temp, bool call_minmax);
    
    void update_with_move(int move);

    void set_num_playout(int num) {
        this->num_playout = num;
    }

    void set_cpuct(int c) {
        this->c_puct = c;
    }

private:

    std::pair<int, int> negamax_no_table(bit_board current_board, int current, int depth, int alpha, int beta) {
        if (current_board.has_winner().first) return std::make_pair(2 * current_board.has_winner().second * current, -1);
        if (depth == 0) {
            return std::make_pair(1, -1);
        }

        std::vector<int> valid_move;
        for (int i = 0 ; i < 7; ++i) {
            if (current_board.can_move(i)) {
                valid_move.push_back(i);
            }
        }

        std::pair<int, int> nextmove = {-2, -1};
        for (auto p : valid_move) {
            auto nxt_board = current_board.duplicate();
            nxt_board.do_move(p);
            auto nxt = negamax_no_table(nxt_board, -current, depth - 1, -beta, -alpha);
            nxt.first *= -1;
            if (nxt.first >= nextmove.first || nextmove.second == -1) {
                nextmove.second = p;
                nextmove.first = nxt.first;
            }

            alpha = std::max(alpha, nextmove.first);

            if (alpha >= beta) {
                break;
            }
        }

        return nextmove;
    }

    std::shared_ptr<policy_value_net> network;
    std::shared_ptr<mcts_node> root;
    int num_playout;
    int c_puct;
};

class mcts_zero : public gameplayer {
public:
    mcts_zero() {
        this->rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
    
    void init(int turn, std::string name, ConfigObject config);

    int winning_move(int previous_move) {
        return -1;
    }
    
    /* 
      @previous_move: integer between 0 and max_col - 1 represents the column the opponent moves, 
                        -1 means the current move is the first move
      @return: a number between 0 and max_col - 1 represents the column of the current player is playing
    */ 
    int play(int previous_move);

    int force_move(int previous_move, int move);

    /*
      @return: the name of the player
    */
    std::string display_name();
    /*
      do something when the game is over, in many cases the player does nothing here
    */
    void game_over(int result);
    /* used for debug, can do nothing */
    void debug();
    /*
        A self-play game, return a tuple of (winner of the game, (board state of each step, position probablity of each step, winning probablity of each step))
    */
    std::tuple<int, std::tuple<std::vector<std::vector<std::vector<std::vector<double>>>>, std::vector<std::vector<double>>, std::vector<int>>> self_play(double temp=1e-3);

    std::tuple<double, double> train_step(std::vector<std::vector<std::vector<std::vector<double>>>> &batch, 
                                        std::vector<std::vector<double>> &mcts_probability, 
                                        std::vector<int> &winner);

    void set_train(ConfigObject config, bool training);

    void save_model(std::string path, std::string optpath);
    
    void reset_player();
protected:
    /*
        return (move, a vector of move probablity)
    */
    std::tuple<int, std::vector<double>> get_action(bit_board board, double temp=1e-3, bool self_play=false);

private:
    
    std::default_random_engine rng = std::default_random_engine {};
    std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(0.0, 1.0);
    double temp, alpha, noise_portion;
    bool is_train, call_minmax;
    mcts_zero_tree mcts;
    std::shared_ptr<policy_value_net> network;
    std::string name;
    bit_board board;
};
