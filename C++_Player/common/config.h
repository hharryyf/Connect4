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

    ConfigObject &Set_dqn_lr(double lr) {
        this->dqn_lr = lr;
        return *this;
    }

    ConfigObject &Set_dqn_decay(double decay) {
        this->dqn_decay = decay;
        return *this;
    }

    ConfigObject &Set_dqn_temp(double tmp) {
        this->dqn_temp = tmp;
        return *this;
    }

    ConfigObject &Set_dirichlet_alpha(double alpha) {
        this->dqn_alpha = alpha;
        return *this;
    }

    ConfigObject &Set_dqn_noise_portion(double noise) {
        this->dqn_noise_portion = noise;
        return *this;
    }

    ConfigObject &Set_reload(bool dqn_reload) {
        this->dqn_reload = dqn_reload;
        return *this;
    }

    ConfigObject &Set_alpha_beta_cache_lost(bool cache) {
        this->alpha_beta_cache_lost = cache;
        return *this;
    }

    ConfigObject &Set_dqn_call_minmax(bool cmn) {
        this->dqn_call_minmax = cmn;
        return *this;
    }

    ConfigObject &Set_file_path(std::string path) {
        this->dqn_file_path = path;
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

    double get_lr() {
        return dqn_lr;
    }

    double get_decay() {
        return dqn_decay;
    }

    double get_temp() {
        return dqn_temp;
    }

    double get_noise_portion() {
        return dqn_noise_portion;
    }

    double get_dirichlet_alpha() {
        return dqn_alpha;
    }

    bool get_dqn_reload() {
        return dqn_reload;
    }

    bool get_alpha_beta_cache_lost() {
        return alpha_beta_cache_lost;
    }

    bool get_dqn_call_minmax() {
        return dqn_call_minmax;
    }

    std::string get_model_path() {
        return dqn_file_path;
    }

private:
    std::string dqn_file_path = "../../model/best_model_rs_13.pt";
    int alpha_beta_max_depth = 11;
    int mcts_play_iteration = 10000;
    int mcts_train_iteration = 500;
    int c_puct = 2;
    double dqn_lr = 0.002;
    double dqn_decay = 0.0001;
    double dqn_temp = 0.001;
    double dqn_noise_portion = 0.25;
    double dqn_alpha = 0.3;
    bool dqn_reload = true;
    bool alpha_beta_cache_lost = true;
    bool dqn_call_minmax = false;
};