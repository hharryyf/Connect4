#include "alphabeta.h"

void alphabeta_player::init(int turn, std::string n) {
    this->player = turn;
    this->name = n;
    this->board.init();
}

std::pair<int, int> alphabeta_player::minimax(int current, int depth, int alpha, int beta) {
    return std::make_pair(0, -1);
}

int alphabeta_player::play(int previous) {
    return -1;
}