#pragma once 
#include "gameplayer.h"
#include "alphabeta_board.h"

class alphabeta_player : public gameplayer {
public:
    void init(int turn, std::string n="Alpha-Beta AI");
    int play(int previous_move);
    std::string display_name() {
        return this->name;
    }
private:
    int player = 0;
    std::string name;
    alphabeta_board board;
    /*
        recursive function
        @return: (evaluation score, move)
        @depth: int, search depth
        @current_player: int, current player
    */
    std::pair<int, int> minimax(int depth, int current_player, int alpha, int beta);
};
