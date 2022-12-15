#include "mcts_zero.h"


std::tuple<std::vector<std::pair<int, double>>, double> policy_value_net::evaluate_position(bit_board &board) {
    std::vector<int> available = board.get_available();
    auto vec_rep = board.get_neural_state();
    auto state = torch::zeros({1, 3, 6, 7});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0 ; j < 6; ++j) {
            for (int k = 0 ; k < 7; ++k) {
                state[0][i][j][k] = vec_rep[i][j][k];
            }   
        }
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(state);
    auto output = this->module.forward(inputs);
    auto policy = output.toTuple()->elements()[0].toTensor();
    policy = torch::exp(policy);
    float score = output.toTuple()->elements()[1].toTensor()[0][0].item<float>();
    std::vector<std::pair<int, double>> move_probability;
    for (auto &valid : available) {
        move_probability.emplace_back(valid, policy[0][valid].item<float>());
    }
    return std::make_tuple(move_probability, score);
}

std::tuple<std::vector<std::vector<double>>, std::vector<double>> policy_value_net::evaluate_batches(std::vector<std::vector<std::vector<std::vector<double>>>> &batch) {
    auto state = torch::zeros({(int) batch.size(), 3, 6, 7});
    for (int b = 0; b < (int) batch.size(); ++b) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0 ; j < 6; ++j) {
                for (int k = 0 ; k < 7; ++k) {
                    state[b][i][j][k] = batch[b][i][j][k];
                }   
            }
        }
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(state);
    auto output = this->module.forward(inputs);
    auto policies = output.toTuple()->elements()[0].toTensor();
    policies = torch::exp(policies);
    auto scores = output.toTuple()->elements()[1].toTensor();
    std::vector<std::vector<double>> policy_ret = std::vector<std::vector<double>>(batch.size(), std::vector<double>(7, 0));
    std::vector<double> value_ret = std::vector<double>(batch.size(), 0);
    for (int i = 0 ; i < (int) batch.size(); ++i) {
        value_ret[i] = scores[i][0].item<float>();
        for (int j = 0 ; j < 7; ++j) {
            policy_ret[i][j] = policies[i][j].item<float>();
        }
    }

    return std::make_tuple(policy_ret, value_ret);
}


std::tuple<double, double> policy_value_net::train_step(std::vector<std::vector<std::vector<std::vector<double>>>> &batch, 
                                        std::vector<std::vector<double>> &mcts_probability, 
                                        std::vector<int> &winner) {
    if (batch.size() != mcts_probability.size() || mcts_probability.size() != winner.size()) {
        printf("batch size = %d, mcts_probability size = %d, winner size = %d not equal!\n", (int) batch.size(), (int) mcts_probability.size(), (int) winner.size());
        exit(1);
    }

    int sz = winner.size();
    auto winner_tensor = torch::ones({sz, 1});
    auto policy_batch = torch::ones({sz, 7});
    for (int i = 0 ; i < sz; ++i) winner_tensor[i][0] = winner[i];
    for (int i = 0 ; i < sz; ++i) {
        for (int j = 0 ; j < 7; ++j) {
            policy_batch[i][j] = mcts_probability[i][j];
        }
    }

    auto state = torch::zeros({(int) batch.size(), 3, 6, 7});
    for (int b = 0; b < (int) batch.size(); ++b) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0 ; j < 6; ++j) {
                for (int k = 0 ; k < 7; ++k) {
                    state[b][i][j][k] = batch[b][i][j][k];
                }   
            }
        }
    }

    this->optimizer->zero_grad();
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(state);
    auto output = this->module.forward(inputs);
    auto log_policy = output.toTuple()->elements()[0].toTensor();
    auto curr_value = output.toTuple()->elements()[1].toTensor();
    auto loss_value = torch::mse_loss(curr_value, winner_tensor);
    auto loss_policy = -torch::mean(torch::sum(log_policy * policy_batch, 1));
    auto loss = loss_value + loss_policy;
    loss.backward();
    this->optimizer->step();
    auto entropy = -torch::mean(torch::sum(torch::exp(log_policy) * log_policy, 1));
    return std::make_tuple(loss.item<float>(), entropy.item<float>());
}

void policy_value_net::save_model(std::string filename) {
    this->module.save(filename);
}

void policy_value_net::set_train() {
    this->module.train();
}

void policy_value_net::set_eval() {
    this->module.eval();
}

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
    this->mcts = mcts_zero_tree(config.get_c_puct(), config.get_mcts_play_iteration());
    if (config.get_dqn_reload()) {
        this->mcts.attach_policy_value_function(this->network);
    }

    this->board = bit_board();
}

int mcts_zero::play(int previous_move) {
    // TODO
    return -1;
}


std::string mcts_zero::display_name() {
    return this->name;
}

std::tuple<int, std::tuple<std::vector<std::vector<std::vector<std::vector<double>>>>, std::vector<std::vector<double>>, std::vector<double>>> mcts_zero::self_play(double temp) {
    // TODO
    std::vector<std::vector<std::vector<std::vector<double>>>> board_states;
    std::vector<std::vector<double>> move_probabilities;
    std::vector<double> winners;
    return std::make_tuple(-1, std::make_tuple(board_states, move_probabilities, winners));
}

std::tuple<int, std::vector<double>> mcts_zero::get_action(bit_board board, double temp) {
    // TODO
    std::vector<double> move_prob;
    return std::make_tuple(-1, move_prob);
}

void mcts_zero::reset_player() {
    this->mcts.update_with_move(-1);
    this->board = bit_board();
}

void mcts_zero::set_train(ConfigObject config, bool training) {
    this->is_train = training;
    if (this->is_train) {
        this->network->set_train();
        this->mcts.set_num_playout(config.get_mcts_train_iteration());
    } else {
        this->network->set_eval();
        this->mcts.set_num_playout(config.get_mcts_play_iteration());
    }
}

void mcts_zero::game_over(int result) {

}

void mcts_zero::debug() {

}

