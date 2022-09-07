#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <stack>
#include <vector>
#include <array>
#include <set>
#include <queue>
#include <cassert>
#include <cmath>
#include <map>
#include <unordered_map>
#include <random>
#include <unordered_set>
#include <bitset>
#include <string>
#include "gameplayer.h"
#include "connect4_board.h"
#include "human_play/humanplayer.h"

int main() {
    connect4_board board;
    board.init();
    std::cout << board.print_board() << std::endl;
    return 0;
}