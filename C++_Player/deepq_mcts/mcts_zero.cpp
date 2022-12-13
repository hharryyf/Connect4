#include "mcts_zero.h"

void mcts_zero::init(int turn, std::string name, ConfigObject config) {
    if (config.get_dqn_reload()) {
        this->network = policy_value_net("../../model/resblock.pt", config.get_lr(), config.get_decay());
    }
    
    this->name = name;
}

int mcts_zero::play(int previous_move) {
    return -1;
}

int mcts_zero::force_play(int position) {
    return -1;
}

std::string mcts_zero::display_name() {
    return this->name;
}

void mcts_zero::game_over(int result) {

}

void mcts_zero::debug() {

}