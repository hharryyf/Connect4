#include "alphabeta.h"

void alphabeta_player::init(int turn, std::string n) {
    this->player = turn;
    this->name = n;
    this->board.init();
}

std::pair<double, int> alphabeta_player::minimax(int current, int depth, double alpha, double beta) {
    // std::cout << "search depth " << depth << std::endl;
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

        return nextmove;
    }
}

int alphabeta_player::play(int previous) {
    if (previous != -1) {
        this->board.update(previous, -this->player);
    }

    auto get_depth = [](int piece) -> int {
        if (piece < 7) {
            return 9;
        } else if (piece < 12) {
            return 11;
        } else if (piece < 18) {
            return 14;
        }

        return 18;
    };

    auto p = minimax(this->player, get_depth(this->board.get_move()), -four, four);
    this->board.update(p.second, this->player);
    std::cout << display_name() << " move in column " << p.second << " heuristic score = " << p.first << std::endl;
    return p.second;
}