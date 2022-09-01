#include "humanplayer.h"

void human_player::init(int turn) {
    this->player = turn;
    this->brd.init();
}

int human_player::play(int previous_move) {
    if (previous_move != -1) {
        // add code here to mark the previous_move on the board

    }

    this->brd.print_bord();
    std::cout << "please input the move " << std::endl;
    int v;
    while (scanf("%d", &v)) {
        if (!this->brd.canplay(v)) {
            std::cout << "a move in column " << v << " is invalid because the column is full" << std::endl;
        } else {
            // add code here to mark the current move v on the board
            return v;
        }
    }

    return -1;
}