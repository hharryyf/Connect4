#pragma once
#include "gameplayer.h"
#include "connect4_board.h"

class human_player : public gameplayer {
public:
    void init(int turn, std::string n, ConfigObject config);
    int winning_move(int previous_move) {
        return -1;
    }
    
    int play(int previous_move);
    int force_move(int previous_move, int move);
    std::string display_name() {
        return this->name;
    }

    void game_over(int result) {
        return;
    }

    void debug() {
        return;
    }
private:
    std::string name;
    int player = 0;
    connect4_board brd;
};