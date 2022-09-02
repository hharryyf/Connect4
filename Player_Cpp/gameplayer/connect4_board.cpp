#include "connect4_board.h"

void board::init() {
    this->move = 0;
    for (int i = 0 ; i < 7; ++i) col[i] = -1;
}

bool board::canplay(int c) {
    return col[c] <= 5;
}

void board::update(int c, int piece) {
    if (piece == 0) {
        a[c][col[c]] = piece;
        --move;
    } else {
        a[c][++col[c]] = piece;
        ++move;
    }
}

static bool check(int x, int y) {
    return x >= 0 && x < 6 && y >= 0 && y < 7;
}

int board::get_status() {
    int i, j;
    for (i = 0 ; i < 6; ++i) {
        for (j = 0 ; j < 7; ++j) {
            if (check(i, j) && check(i+1, j) && check(i+2, j) && check(i+3, j)
            && abs(a[i][j] + a[i+1][j] + a[i+2][j] + a[i+3][j]) == 4) {
                return a[i][j] + a[i+1][j] + a[i+2][j] + a[i+3][j] > 0 ? 1 : -1;
            }

            if (check(i, j) && check(i+1, j+1) && check(i+2, j+2) && check(i+3, j+3)
            && abs(a[i][j] + a[i+1][j+1] + a[i+2][j+2] + a[i+3][j+3]) == 4) {
                return a[i][j] + a[i+1][j+1] + a[i+2][j+2] + a[i+3][j+3] > 0 ? 1 : -1;
            }

            if (check(i, j) && check(i+1, j-1) && check(i+2, j-2) && check(i+3, j-3)
            && abs(a[i][j] + a[i+1][j-1] + a[i+2][j-2] + a[i+3][j-3]) == 4) {
                return a[i][j] + a[i+1][j-1] + a[i+2][j-2] + a[i+3][j-3] > 0 ? 1 : -1;
            } 
        }
    }
    if (move == 42) return 0;
    return 2;
}

std::string board::print_board() {
    std::string ret;
    for (int i = 5; i >= 0; --i) {
        for (int j = 0 ; j < 7; ++j) {
            if (a[i][j] == 0) {
                ret.append(".");
            } else if (a[i][j] == 1) {
                ret.append("X");
            } else {
                ret.append("O");
            }
            
            if (j != 6) {
                ret.append("|");
            } else {
                ret.append("\n");
            }
        }

        if (i != 0) {
            for (int j = 0 ; j < 7; ++j) {
                ret.append("_");
                if (j != 6) {
                    ret.append(" ");
                } else {
                    ret.append("\n");
                }
            }
        }
    }
    
    return ret;
}