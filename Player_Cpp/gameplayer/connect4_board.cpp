#include "connect4_board.h"

void board::init() {

}

bool board::canplay(int col) {
    return true;
}

void board::update(int r, int c, int piece) {

}

int board::get_status() {
    return 2;
}

std::pair<int, int> board::get_connect_3() {
    return std::make_pair(0, 0);
}

std::pair<int, int> board::get_connect_3_weak() {
    return std::make_pair(0, 0);
}

std::pair<int, int> board::get_connect_2() {
    return std::make_pair(0, 0);
}

void board::print_bord() {
    
}