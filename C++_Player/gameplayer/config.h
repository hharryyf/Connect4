#pragma once
#include <iostream>
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