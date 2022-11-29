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
    mcts_node(std::shared_ptr<mcts_node> parent=nullptr, double prior=1.0) {
        this->parent = parent;
        this->prior = prior;
    }

    void expansion(std::vector<std::pair<int, double>> &action_prior) {
        for (auto &p : action_prior) {
            int move = p.first;
            double prob = p.second;
            if (children.find(move) == children.end()) {
                children[move] = std::make_shared<mcts_node>(mcts_node(std::make_shared<mcts_node>(*this), prob));
            }
        }
    }

    std::pair<int, std::shared_ptr<mcts_node>> selection(double c_puct) {
        assert (!is_leaf());
        std::pair<int, std::shared_ptr<mcts_node>> ret = {-1, nullptr};
        for (auto &iter : this->children) {
            if (ret.second == nullptr) ret = iter;
            if (ret.second->get_value(c_puct) < iter.second->get_value(c_puct)) ret = iter;
        }

        return ret;
    }

    int select_move() {
        int move = -1, vis = 0;
        for (auto &iter : this->children) {
            if (move == -1) move = iter.first, vis = iter.second->N;
            if (vis < iter.second->N) vis = iter.second->N, move = iter.first;
        }

        return move;
    }

    double get_value(double c_puct) {
        this->U = c_puct * this->prior * sqrt(this->parent->N) / (1 + this->N);
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
        return this->children.empty();
    }

    std::shared_ptr<mcts_node> get_children(int move) {
        if (this->children.find(move) != this->children.end()) return this->children[move];
        return nullptr;
    }

    void set_parent(std::shared_ptr<mcts_node> new_parent) {
        this->parent = new_parent;
    }

private:
    double prior = 1.0, U = 0.0, Q = 0.0;
    int N = 0;
    std::map<int, std::shared_ptr<mcts_node>> children;    
    std::shared_ptr<mcts_node> parent;
};