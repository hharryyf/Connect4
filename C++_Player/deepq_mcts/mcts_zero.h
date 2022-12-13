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
#include "connect4_board.h"
#include "bit_board.h"
#include "gameplayer.h"
#include "mcts_node.h"
#pragma warning(push, 0)
#include <torch/torch.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/script.h>
#pragma warning(pop)

class policy_value_net {
public:
    policy_value_net(std::string filename="../../model/resblock.pt", double lr=0.002, double decay=0.0001) {
        this->module = torch::jit::load(filename);
        for (const auto& params : module.parameters()) {
	        this->parameters.push_back(params);
        }
        
        this->optimizer = std::make_shared<torch::optim::Adam>(this->parameters, torch::optim::AdamOptions(lr).weight_decay(decay));
    }
    
    void evaluate_position(bit_board &board) {
        
    }

    void evaluate_batches(std::vector<std::vector<std::vector<std::vector<double>>>> &batch) {

    }

    void train_step(std::vector<std::vector<std::vector<std::vector<double>>>> &batch, std::vector<std::vector<double>> &mcts_probability, std::vector<double> &winner) {
        if (batch.size() != mcts_probability.size() || mcts_probability.size() != winner.size()) {
            printf("batch size = %d, mcts_probability size = %d, winner size = %d not equal!\n", (int) batch.size(), (int) mcts_probability.size(), (int) winner.size());
            exit(1);
        }
    }

    void save_model(std::string filename) {
        this->module.save(filename);
    }

    void set_train() {
        this->module.train();
    }

    void set_eval() {
        this->module.eval();
    }

private:
    torch::jit::Module module;
    std::vector<at::Tensor> parameters;
    std::shared_ptr<torch::optim::Optimizer> optimizer;
};

class mcts_zero : public gameplayer {
public:
    mcts_zero() {

    }
    
    void init(int turn, std::string name, ConfigObject config);
    /* 
      @previous_move: integer between 0 and max_col - 1 represents the column the opponent moves, 
                        -1 means the current move is the first move
      @return: a number between 0 and max_col - 1 represents the column of the current player is playing
    */ 
    int play(int previous_move);
    /*
      Must take a move at some position
      return value is equal to position
    */
    int force_play(int position) = 0;
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

private:
    policy_value_net network;
    std::string name;
};