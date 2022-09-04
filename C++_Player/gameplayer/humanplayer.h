#pragma once
#include "gameplayer.h"
#include "connect4_board.h"

class human_player : public gameplayer {
public:
    void init(int turn, std::string n="Human Player");
    int play(int previous_move);
    std::string display_name() {
        return this->name;
    }

    void game_over() {
        return;
    }
private:
    std::string name;
    int player = 0;
    connect4_board brd;
};