#pragma once 
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
#include "config.h"
#include "gameplayer.h"

class combine_player : public gameplayer {
public:
    combine_player() {
        this->num_move = 0;
    }
    void init(int turn, std::string name, ConfigObject config); 
    int play(int previous_move);
    int force_move(int previous_move, int move);
    std::string display_name();
    void set_players(gameplayer *g1, gameplayer *g2);
    void game_over(int result);
    void debug();
private:
    gameplayer *g1;
    gameplayer *g2;
    std::string name;
    int num_move;
};