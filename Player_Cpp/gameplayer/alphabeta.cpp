#include "alphabeta.h"

void alphabeta_player::init(int turn) {
    this->player = turn;
    this->brd.init();
}

int alphabeta_player::play(int previous) {
    return -1;
}