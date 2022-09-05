#pragma once
#include <iostream>
#include <string>
/*
    configuration for the alpha-beta player
*/

namespace AlphaBetaConfig {

    // discount means how many steps we need to fill this node
    const double discount = 0.9;
    // value for 4 in a row
    const double four = 100000000.0;
    // value for 3 in a row
    const double three = 200000.0;
    // value for 2 in a row
    const double two = 50000.0;
    // value for 1 in a row
    const double one = 1000.0;
    // some "enum" values
    const int ROW = 0, COL = 1, DIAG = 2, ANTIDIAG = 3;
    const int max_depth = 11;

    const std::string cache_file = "memo.txt";

    const int max_cache = 300000;
}