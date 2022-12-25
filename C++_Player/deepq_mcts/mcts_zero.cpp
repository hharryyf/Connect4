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
    for (int i = 0 ; i < sz; ++i) winner_tensor[i][0] = 1.0 * winner[i];
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

void mcts_zero_tree::playout(bit_board board, bool call_minmax) {
    auto curr = this->root;
    while (curr != nullptr) {
        if (curr->is_leaf()) break;
        auto mpn = curr->selection(this->c_puct);
        board.do_move(mpn.first);
        curr = mpn.second;
    }

    auto eval = this->network->evaluate_position(board);
    auto action_probablity = std::get<0>(eval);
    auto reward = std::get<1>(eval);
    auto result = board.has_winner();
    if (result.first) {
        if (result.second == 0) {
            reward = 0;
        } else {
            reward = board.get_current_player() == result.second ? 1 : -1;
        }
    } else {
        if (call_minmax) {
            auto best = this->negamax_no_table(board, board.get_current_player(), 5, -2, 2);
            if (best.first == 2 || best.first == 0 || best.first == -2) {
                // printf("we fix the reward from %lf to %lf\n", reward, 0.5 * best.first);
                reward = 0.5 * best.first;
            }
            if (best.first == 2 || best.first == 0) {
                for (auto &prob : action_probablity) {
                    if (prob.first == best.second) {
                        prob.second = 1.0;
                    } else {
                        prob.second = 0.0;
                    }
                }
            }
        }

        curr->expansion(action_probablity);
    }

    curr->update_recursive(-reward);
}

std::tuple<std::vector<int>, std::vector<double>> mcts_zero_tree::get_move_probability(bit_board board, double temp, bool call_minmax) {
    for (int i = 0 ; i < this->num_playout; ++i) {
        this->playout(board, call_minmax);
    }


    return this->root->get_action_probability(temp);
}

void mcts_zero_tree::update_with_move(int move) {
    auto nxt = this->root->get_children(move);
    if (nxt != nullptr) {
        this->root = nxt;
        this->root->set_parent(nullptr);
    } else {
        this->root = std::make_shared<mcts_node>(mcts_node(nullptr, 1.0));
    }
}

void mcts_zero::init(int turn, std::string name, ConfigObject config) {
    if (config.get_dqn_reload()) {
        this->network = std::make_shared<policy_value_net>(config.get_model_path(), config.get_lr(), config.get_decay());
    }
    
    this->name = name;
    this->mcts = mcts_zero_tree(config.get_c_puct(), config.get_mcts_play_iteration());
    this->mcts.attach_policy_value_function(this->network);
    this->temp = config.get_temp();
    this->alpha = config.get_dirichlet_alpha();
    this->noise_portion = config.get_noise_portion();
    this->call_minmax = config.get_dqn_call_minmax();
    this->board = bit_board();
    //printf("reload? = %d\n", config.get_dqn_reload());
}

int mcts_zero::play(int previous_move) {
    if (this->board.game_end()) {
        printf("cannot play when the game ends!\n");
        exit(1);
    }

    if (previous_move != -1) { 
        this->board.do_move(previous_move);
    }

    
    if (this->board.game_end()) {
        printf("cannot play when the game ends!\n");
        exit(1);
    }

    if (previous_move == -1) {
        // we play in the center for the first move
        // this uses some expert knowledge
        this->board.do_move(3);
        return 3;
    }

    // general case
    int move = std::get<0>(this->get_action(this->board, this->temp, false));
    this->board.do_move(move);
    // no need to call: this->mcts.update_with_move(-1) because it has already been called in get_action
    return move;
}

int mcts_zero::force_move(int previous_move, int move) {
    if (previous_move != -1) { 
        this->board.do_move(previous_move);
    }

    this->board.do_move(move);
    return move;
}

std::string mcts_zero::display_name() {
    return this->name;
}

std::tuple<int, std::tuple<std::vector<std::vector<std::vector<std::vector<double>>>>, std::vector<std::vector<double>>, std::vector<int>>> mcts_zero::self_play(double temp) {
    int winner = 0;
    std::vector<std::vector<std::vector<std::vector<double>>>> board_states;
    std::vector<std::vector<double>> move_probabilities;
    std::vector<int> current_players;
    std::vector<int> winners;
    this->reset_player();
    while (1) {
        auto act = this->get_action(this->board, temp, true);
        int move = std::get<0>(act);
        printf("self-play move = %d\n", move);
        std::vector<double> move_prob = std::get<1>(act);
        board_states.push_back(this->board.get_neural_state());
        move_probabilities.push_back(move_prob);
        current_players.push_back(this->board.get_current_player());
        this->board.do_move(move);
        auto result = this->board.has_winner();
        if (result.first) {
            winner = result.second;
            winners = std::vector<int>(current_players.size(), 0);
            if (winner != 0) {
                for (int i = 0 ; i < (int) winners.size(); ++i) {
                    winners[i] = current_players[i] == winner ? 1 : -1;
                }
            }
            if (result.second == 0) {
                printf("Draw!\n");
            } else {
                printf("Winner is player %d\n", result.second);
            }

            this->reset_player();
            break;
        }

    }
    return std::make_tuple(winner, std::make_tuple(board_states, move_probabilities, winners));
}

std::tuple<int, std::vector<double>> mcts_zero::get_action(bit_board board, double temp, bool self_play) {
    // TODO
    int move = -1;
    std::vector<double> move_prob = std::vector<double>(7, 0);
    if (!this->board.has_winner().first) {
        auto act = this->mcts.get_move_probability(board, temp, this->call_minmax);
        auto valid_action = std::get<0>(act);
        auto act_prob = std::get<1>(act);
        for (int i = 0 ; i < (int) valid_action.size(); ++i) {
            move_prob[valid_action[i]] = act_prob[i];
        }

        dirichlet_distribution<std::default_random_engine> d(std::vector<double>(valid_action.size(), this->alpha));
        std::vector<double> noise = d(this->rng);
        auto sample_prob = [&](std::vector<double> vc) -> int {
            if (vc.empty()) {
                printf("Error try to sample from an empty distribution!");
                exit(1);
            }

            double sm = 0;
            for (auto &p : vc) sm = sm + p;
            for (auto &p : vc) p /= sm;
            for (int i = 1; i < (int) vc.size(); ++i) vc[i] += vc[i-1];
            double g = this->distribution(this->rng);
            for (int i = 0 ; i < (int) vc.size(); ++i) {
                if (g <= vc[i]) return i;
            }

            return (int) vc.size() - 1;
        };

        if (self_play) {    
            // choose move based on the distribution of act_prob, add some dirichlet noise
            for (int i = 0 ; i < (int) act_prob.size(); ++i) {
                act_prob[i] = (1.0 - this->noise_portion) * act_prob[i] + this->noise_portion * noise[i];
            }
            
            move = valid_action[sample_prob(act_prob)];
            this->mcts.update_with_move(move);
        } else {
            // choose move based on the distribution of act_prob
            move = valid_action[sample_prob(act_prob)];
            this->mcts.update_with_move(-1);
        }
    } else {
        printf("Warning! Call get action for end game position!\n");
    }

    return std::make_tuple(move, move_prob);
}

void mcts_zero::reset_player() {
    this->mcts.update_with_move(-1);
    this->board = bit_board();
}

void mcts_zero::set_train(ConfigObject config, bool training) {
    this->is_train = training;
    this->call_minmax = config.get_dqn_call_minmax();
    if (this->is_train) {
        this->network->set_train();
        this->mcts.set_num_playout(config.get_mcts_train_iteration());
    } else {
        this->network->set_eval();
        this->mcts.set_num_playout(config.get_mcts_play_iteration());
    }
}

std::tuple<double, double> mcts_zero::train_step(std::vector<std::vector<std::vector<std::vector<double>>>> &batch, 
                                        std::vector<std::vector<double>> &mcts_probability, 
                                        std::vector<int> &winner) {
    return this->network->train_step(batch, mcts_probability, winner);
}

void mcts_zero::save_model(std::string path) {
    this->network->save_model(path);
}

void mcts_zero::game_over(int result) {

}

void mcts_zero::debug() {

}
