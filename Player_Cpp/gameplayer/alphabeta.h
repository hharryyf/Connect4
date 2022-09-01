#pragma once 
#include "gameplayer.h"
#include "alphabeta_board.h"

class alphabeta_player : public gameplayer {
public:
    void init(int turn);
    int play(int previous_move);
private:
    int player = 0;
    alphabeta_board board;
    /*
        recursive function
        @return: (evaluation score, move)
        @depth: int, search depth
        @current_player: int, current player
    */
    std::pair<int, int> minimax(int depth, int current_player, int alpha, int beta);
};
