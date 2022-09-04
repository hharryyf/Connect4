#include "alphabeta.h"

void alphabeta_player::init(int turn, std::string n) {
    this->player = turn;
    this->name = n;
    this->board.init();
}

std::pair<double, int> alphabeta_player::minimax(int current, int depth, double alpha, double beta) {
    auto status = this->board.status();
    if (status != 2) return std::make_pair(four * status, -1);
    if (depth == 0) return std::make_pair(this->board.heuristic(), -1);
    std::vector<int> valid_move;
    for (int i = 0 ; i < 7; ++i) {
        if (this->board.can_move(i)) {
            valid_move.push_back(i);
        }
    }

    std::shuffle(valid_move.begin(), valid_move.end(), rng);

    for (auto p : valid_move) {
        if (this->board.killmove(p, current)) {
            return std::make_pair(four * current, p);
        }
    }

    auto cache = this->board.get_cached_move();
    if (cache.second != -1) {
        //std::cout << "reach cache at depth " << depth << std::endl;
        return cache;
    }

    if (current == 1) {
        std::pair<double, int> nextmove = {-four, -1};
        for (auto p : valid_move) {
            this->board.update(p, current);
            auto nxt = minimax(-current, depth - 1, alpha, beta);
            this->board.update(p, 0);
            if (nxt.first >= nextmove.first || nextmove.second == -1) {
                nextmove.second = p;
                nextmove.first = nxt.first;
            }

            alpha = std::max(alpha, nextmove.first);

            if (alpha >= beta) {
                return nextmove;
            }
        }

        if (nextmove.first == four || nextmove.second == -four) {
            this->board.catche_state(nextmove);
        }

        return nextmove;
    } else {
        std::pair<double, int> nextmove = {four, -1};
        for (auto p : valid_move) {
            this->board.update(p, current);
            auto nxt = minimax(-current, depth - 1, alpha, beta);
            this->board.update(p, 0);
            if (nxt.first <= nextmove.first || nextmove.second == -1) {
                nextmove.second = p;
                nextmove.first = nxt.first;
            }

            beta = std::min(beta, nextmove.first);

            if (alpha >= beta) {
                return nextmove;
            }
        }

        if (nextmove.first == four || nextmove.second == -four) {
            this->board.catche_state(nextmove);
        }
        return nextmove;
    }
}

int alphabeta_player::play(int previous) {
    if (previous != -1) {
        this->board.update(previous, -this->player);
    }

    auto get_depth = [](int piece) -> int {
        if (piece < 12) {
            return max_depth;
        } else if (piece < 16) {
            return 4 + max_depth;
        }

        return 8 + max_depth;
    };

    int d = get_depth(this->board.get_move());
    auto p = minimax(this->player, d, -four, four);
    this->board.update(p.second, this->player);
    std::cout << display_name() << " search depth: " << d << " score: " << p.first << std::endl;
    // this->board.debug();
    return p.second;
}
