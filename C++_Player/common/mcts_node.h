#pragma once
#include <cstdio>
#include <iostream>
#include <memory>
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

class mcts_node {
public:
    mcts_node(mcts_node* parent=nullptr, double prior=1.0) {
        this->parent = parent;
        this->prior = prior;
    }

    void expansion(std::vector<std::pair<int, double>> &action_prior) {
        for (auto &p : action_prior) {
            int move = p.first;
            double prob = p.second;
            if (this->children[move] == nullptr) {
                children[move] = std::make_shared<mcts_node>(nullptr, prob);
                children[move]->set_parent(this);
            }
        }
    }

    std::pair<int, std::shared_ptr<mcts_node>> selection(double c_puct) {
        std::pair<int, std::shared_ptr<mcts_node>> ret = {-1, nullptr};
        for (int i = 0 ; i < 7; ++i) {
            if (this->children[i] != nullptr) {
                if (ret.second == nullptr) ret.second = this->children[i], ret.first = i;
                if (ret.second->get_value(c_puct) < this->children[i]->get_value(c_puct)) {
                    ret.second = this->children[i];
                    ret.first = i;
                }
            }
        }



        return ret;
    }

    int select_move() {
        int move = -1, vis = 0;
        for (int i = 0 ; i < 7; ++i) {
            if (this->children[i] != nullptr) {
                if (move == -1) move = i, vis = this->children[i]->N;
                if (vis < this->children[i]->N) vis = this->children[i]->N, move = i;
            }
        }

        return move;
    }

    double get_value(double c_puct) {
        this->U = c_puct * this->prior * sqrt(1.0 * this->parent->N) / (1 + this->N);
        return this->U + this->Q;
    }

    void update(double val) {
        this->N = this->N + 1;
        this->Q += (val - this->Q) / this->N;
    }

    void update_recursive(double val) {
        if (this->parent != nullptr) this->parent->update_recursive(-val);
        update(val);
    }

    bool is_leaf() {
        for (int i = 0 ; i < 7; ++i) {
            if (this->children[i] != nullptr) return false;
        }

        return true;
    }

    std::shared_ptr<mcts_node> get_children(int move) {
        if (move >= 0 && move <= 6 && this->children[move] != nullptr) return this->children[move];
        return nullptr;
    }

    void set_parent(mcts_node* new_parent) {
        this->parent = new_parent;
    }

    void debug() {
        for (int i = 0 ; i < 7; ++i) {
            if (this->children[i] != nullptr) {
                printf("move = %d, visit count = %d, q_value = %.6f | ", i, this->children[i]->N, this->children[i]->Q);
            }
        }
        printf("\n");
    }

private:
    double prior = 1.0, U = 0.0, Q = 0.0;
    int N = 0;
    std::shared_ptr<mcts_node> children[7];    
    mcts_node* parent;
};