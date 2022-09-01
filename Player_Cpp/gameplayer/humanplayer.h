#pragma once
#include "gameplayer.h"
#include "connect4_board.h"

class human_player : public gameplayer {
public:
    void init(int turn);
    int play(int previous_move);
private:
    int player = 0;
    board brd;
};