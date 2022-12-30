#pragma once 
#include "gameplayer.h"
#include "alphabeta_board.h"

class alphabeta_player : public gameplayer {
public:
    alphabeta_player(bool cache_result=false) : cache_result(cache_result) {}

    void init(int turn, std::string n, ConfigObject config);
    
    int winning_move(int previous_move);

    int play(int previous_move);
    
    int force_move(int previous_move, int move);

    std::string display_name() {
        return this->name;
    }

    void game_over(int result);

    void debug() {
        std::cout << this->name << " play turn " << this->player << std::endl;
        this->board.debug();
    }
protected:
    /*
        recursive function
        @return: (evaluation score with respect to the first player, move)
        @depth: int, search depth
        @current_player: int, current player
    */
    std::pair<double, int> minimax(int current_player, int depth, double alpha, double beta);
    /*
        recursive function
        @return: (evaluation score with respect to the current player, move)
        @depth: int, search depth
        @current_player: int, current player
    */
    std::pair<double, int> negamax(int current_player, int depth, double alpha, double beta);

    std::pair<double, int> negamax_no_table(int current_player, int depth, double alpha, double beta);
    
private:
    bool cache_lost = true;
    std::default_random_engine rng = std::default_random_engine {};
    int player = 0;
    int max_depth = 11;
    bool cache_result = false;
    std::string name;
    alphabeta_board board;
};
