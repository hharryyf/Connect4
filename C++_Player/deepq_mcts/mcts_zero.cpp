#include "mcts_zero.h"


void mcts_zero_tree::playout(bit_board board) {

}

std::tuple<std::vector<int>, std::vector<double>> mcts_zero_tree::get_move_probability(bit_board board, double temp) {
    std::vector<int> valid_move;
    std::vector<double> move_probability;
    return std::make_tuple(valid_move, move_probability);
}

void mcts_zero_tree::update_with_move(int move) {

}

void mcts_zero::init(int turn, std::string name, ConfigObject config) {
    if (config.get_dqn_reload()) {
        this->network = std::make_shared<policy_value_net>("../../model/resblock.pt", config.get_lr(), config.get_decay());
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

std::tuple<int, std::tuple<std::vector<std::vector<std::vector<std::vector<double>>>>, std::vector<std::vector<double>>, std::vector<double>>> mcts_zero::self_play(double temp) {
    std::vector<std::vector<std::vector<std::vector<double>>>> board_states;
    std::vector<std::vector<double>> move_probabilities;
    std::vector<double> winners;
    return std::make_tuple(-1, std::make_tuple(board_states, move_probabilities, winners));
}

std::tuple<int, std::vector<double>> mcts_zero::get_action(bit_board board, double temp) {
    std::vector<double> move_prob;
    return std::make_tuple(-1, move_prob);
}

void mcts_zero::reset_player() {

}

void mcts_zero::set_train(bool is_train) {
    
}

