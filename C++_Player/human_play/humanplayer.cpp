#include "humanplayer.h"

void human_player::init(int turn, std::string n, ConfigObject config) {
    this->player = turn;
    this->name = n;
    this->brd.init();
}

int human_player::play(int previous_move) {
    if (previous_move != -1) {
        // add code here to mark the previous_move on the board
        this->brd.update(previous_move, -this->player);
    }

    // std::cout << this->brd.print_board() << std::endl;
    this->brd.show_board();
    std::cout << "please input the move column: ";
    int v;
    while (scanf("%d", &v)) {
        v--;
        if (!this->brd.canplay(v)) {
            std::cout << "a move in column " << v << " is invalid" << std::endl;
        } else {
            // add code here to mark the current move v on the board
            this->brd.update(v, this->player);
            return v;
        }
    }

    return -1;
}

int human_player::force_move(int previous_move, int move) {
    if (previous_move != -1) {
        // add code here to mark the previous_move on the board
        this->brd.update(previous_move, -this->player);
    }

    this->brd.update(move, this->player);

    return move;
}