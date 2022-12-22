#include "alphabeta.h"

void alphabeta_player::init(int turn, std::string n, ConfigObject config) {
    this->player = turn;
    this->name = n;
    this->board.init();
    this->max_depth = config.get_alpha_beta_max_depth();
    this->cache_lost = config.get_alpha_beta_cache_lost();
}

std::pair<double, int> alphabeta_player::negamax(int current, int depth, double alpha, double beta) {
    double alphaorg = alpha;
    auto status = this->board.status();
    if (status != 2) return std::make_pair(AlphaBetaConfig::four * status * current, AlphaBetaConfig::INVALID);
    auto entry = this->board.get_cached_move();
    if (entry.second != AlphaBetaConfig::INVALID) {
        if (entry.second != AlphaBetaConfig::LOWER_ID && entry.second != AlphaBetaConfig::UPPER_ID) {
            return entry;
        }

        if (entry.second == AlphaBetaConfig::LOWER_ID) {
            alpha = std::max(alpha, entry.first);
        } else if (entry.second == AlphaBetaConfig::UPPER_ID) {
            beta = std::min(beta, entry.first);
        } 
        
        if (alpha >= beta) {
            return entry;
        }
    }

    std::vector<int> valid_move;
    for (int i = 0 ; i < 7; ++i) {
        if (this->board.can_move(i)) {
            valid_move.push_back(i);
        }
    }

    std::shuffle(valid_move.begin(), valid_move.end(), rng);

    for (auto p : valid_move) {
        if (this->board.killmove(p, current)) {
            return std::make_pair(AlphaBetaConfig::four, p);
        }
    }

    if (depth == 0) {
        return std::make_pair(this->board.heuristic() * current, AlphaBetaConfig::INVALID);
    }

    //printf("enter general case\n");
    std::pair<double, int> nextmove = {-AlphaBetaConfig::four, AlphaBetaConfig::INVALID};
    for (auto p : valid_move) {
        this->board.update(p, current);
        auto nxt = negamax(-current, depth - 1, -beta, -alpha);
        this->board.update(p, 0);
        nxt.first *= -1;
        if (nxt.first >= nextmove.first || nextmove.second == AlphaBetaConfig::INVALID) {
            nextmove.second = p;
            nextmove.first = nxt.first;
        }

        alpha = std::max(alpha, nextmove.first);

        if (alpha >= beta) {
            break;
        }
    }

    entry.first = nextmove.first;
    if ((entry.first == AlphaBetaConfig::four && depth >= AlphaBetaConfig::meaningful_depth)) {
        entry.second = nextmove.second;
        this->board.cache_state(entry, AlphaBetaConfig::ENDGAME);
    }

    if (nextmove.first <= alphaorg) {
        entry.second = AlphaBetaConfig::UPPER_ID;
        this->board.cache_state(entry, AlphaBetaConfig::UPPER);
    } else if (nextmove.first >= beta) {
        entry.second = AlphaBetaConfig::LOWER_ID;
        this->board.cache_state(entry, AlphaBetaConfig::LOWER);
    } else {
        entry.second = nextmove.second;
        this->board.cache_state(entry, AlphaBetaConfig::EXACT);
    }

    //printf("score = %.2lf move = %d\n", nextmove.first, nextmove.second);
    return nextmove;
}

std::pair<double, int> alphabeta_player::negamax_no_table(int current, int depth, double alpha, double beta) {
    auto status = this->board.status();
    if (status != 2) return std::make_pair(AlphaBetaConfig::four * status * current, AlphaBetaConfig::INVALID);
    if (depth == 0) {
        return std::make_pair(this->board.heuristic() * current, AlphaBetaConfig::INVALID);
    }

    std::vector<int> valid_move;
    for (int i = 0 ; i < 7; ++i) {
        if (this->board.can_move(i)) {
            valid_move.push_back(i);
        }
    }

    std::shuffle(valid_move.begin(), valid_move.end(), rng);

    for (auto p : valid_move) {
        if (this->board.killmove(p, current)) {
            return std::make_pair(AlphaBetaConfig::four, p);
        }
    }

    std::pair<double, int> nextmove = {-AlphaBetaConfig::four, AlphaBetaConfig::INVALID};
    for (auto p : valid_move) {
        this->board.update(p, current);
        auto nxt = negamax_no_table(-current, depth - 1, -beta, -alpha);
        this->board.update(p, 0);
        nxt.first *= -1;
        if (nxt.first >= nextmove.first || nextmove.second == AlphaBetaConfig::INVALID) {
            nextmove.second = p;
            nextmove.first = nxt.first;
        }

        alpha = std::max(alpha, nextmove.first);

        if (alpha >= beta) {
            break;
        }
    }

    return nextmove;
}

std::pair<double, int> alphabeta_player::minimax(int current, int depth, double alpha, double beta) {
    auto status = this->board.status();
    if (status != 2) return std::make_pair(AlphaBetaConfig::four * status, AlphaBetaConfig::INVALID);
    if (depth == 0) return std::make_pair(this->board.heuristic(), AlphaBetaConfig::INVALID);
    std::vector<int> valid_move;
    for (int i = 0 ; i < 7; ++i) {
        if (this->board.can_move(i)) {
            valid_move.push_back(i);
        }
    }

    std::shuffle(valid_move.begin(), valid_move.end(), rng);

    for (auto p : valid_move) {
        if (this->board.killmove(p, current)) {
            return std::make_pair(AlphaBetaConfig::four * current, p);
        }
    }

    auto cache = this->board.get_cached_move();
    if (cache.second != -1) {
        return cache;
    }

    if (current == 1) {
        std::pair<double, int> nextmove = {-AlphaBetaConfig::four, AlphaBetaConfig::INVALID};
        for (auto p : valid_move) {
            this->board.update(p, current);
            auto nxt = minimax(-current, depth - 1, alpha, beta);
            this->board.update(p, 0);
            if (nxt.first >= nextmove.first || nextmove.second == AlphaBetaConfig::INVALID) {
                nextmove.second = p;
                nextmove.first = nxt.first;
            }

            alpha = std::max(alpha, nextmove.first);

            if (alpha >= beta) {
                break;
            }
        }

        if ((nextmove.first == AlphaBetaConfig::four || nextmove.first == -AlphaBetaConfig::four)) {
            if (depth >= AlphaBetaConfig::meaningful_depth) {
                this->board.cache_state(nextmove, AlphaBetaConfig::ENDGAME);
            } else {
                this->board.cache_state(nextmove, AlphaBetaConfig::EXACT);
            }
        }

        return nextmove;
    } else {
        std::pair<double, int> nextmove = {AlphaBetaConfig::four, AlphaBetaConfig::INVALID};
        for (auto p : valid_move) {
            this->board.update(p, current);
            auto nxt = minimax(-current, depth - 1, alpha, beta);
            this->board.update(p, 0);
            if (nxt.first <= nextmove.first || nextmove.second == AlphaBetaConfig::INVALID) {
                nextmove.second = p;
                nextmove.first = nxt.first;
            }

            beta = std::min(beta, nextmove.first);

            if (alpha >= beta) {
                break;
            }
        }
        
        if ((nextmove.first == AlphaBetaConfig::four || nextmove.first == -AlphaBetaConfig::four)) {
            if (depth >= AlphaBetaConfig::meaningful_depth) {
                this->board.cache_state(nextmove, AlphaBetaConfig::ENDGAME);
            } else {
                this->board.cache_state(nextmove, AlphaBetaConfig::EXACT);
            }
        }

        return nextmove;
    }
}

int alphabeta_player::play(int previous) {
    if (previous != -1) {
        this->board.update(previous, -this->player);
    }

    auto get_depth = [&](int piece) -> int {
        if (piece < 8) {
            return this->max_depth;
        } else if (piece < 14) {
            return 2 + this->max_depth;
        } else if (piece < 16) {
            return 4 + this->max_depth;
        } else if (piece < 20) {
            return 8 + this->max_depth;
        }

        return 10 + this->max_depth;
    };

    this->board.clear_middle_game_cache();
    int d = get_depth(this->board.get_move());
    //auto p = minimax(this->player, d, -AlphaBetaConfig::four, AlphaBetaConfig::four);
    auto p = negamax(this->player, d, -AlphaBetaConfig::four, AlphaBetaConfig::four);
    if (p.first == -AlphaBetaConfig::four) {
        auto q = negamax_no_table(this->player, 3, -AlphaBetaConfig::four, AlphaBetaConfig::four);
        p.second = q.second;
    }
    std::cout << display_name() << " search depth: " << d << " score: " << p.first * this->player << std::endl;
    this->board.update(p.second, this->player);
    std::cout << "cache size: " << this->board.get_cache_size() << std::endl;
    return p.second;
}

int alphabeta_player::force_move(int previous_move, int move) {
    if (previous_move != -1) {
        this->board.update(previous_move, -this->player);
    }

    this->board.update(move, this->player);

    return move;
}

void alphabeta_player::game_over(int result) {
    if (result == -1) {
        if (this->cache_lost) {
            this->board.store_cache();
        }
    }
}