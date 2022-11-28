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
#include <iterator>

class connect4_board {
public:
    // initialize the board
    void init() {
      this->move = 0;
      for (int i = 0 ; i < 7; ++i) col[i] = -1;
      for (int i = 0 ; i < 6; ++i) {
          for (int j = 0 ; j < 7; ++j) {
              a[i][j] = 0;
          }
      }
    }

    // if we can play a move
    bool canplay(int c) {
      return (col[c] < 5) && (c >= 0) && (c < 7);
    }
    // update column c with piece
    void update(int c, int piece) {
        if (piece == 0) {
          a[col[c]][c] = piece;
          col[c]--;
          --move;
        } else {
            ++col[c];
            a[col[c]][c] = piece;
            ++move;
        }
    }
    /* return the status of the board
        -1 means a win for the maximizer
        0 means a draw
        1 means a win for the minimizer
        2 means unknown 
    */
    int get_status() {
        int i, j;
        auto check = [](int x, int y) -> bool {
            return x >= 0 && x < 6 && y >= 0 && y < 7;
        };

        for (i = 0 ; i < 6; ++i) {
            for (j = 0 ; j < 7; ++j) {
                if (check(i, j) && check(i+1, j) && check(i+2, j) && check(i+3, j)
                && abs(a[i][j] + a[i+1][j] + a[i+2][j] + a[i+3][j]) == 4) {
                    return a[i][j] + a[i+1][j] + a[i+2][j] + a[i+3][j] > 0 ? 1 : -1;
                }

                if (check(i, j) && check(i, j+1) && check(i, j+2) && check(i, j+3)
                && abs(a[i][j] + a[i][j+1] + a[i][j+2] + a[i][j+3]) == 4) {
                    return a[i][j] + a[i][j+1] + a[i][j+2] + a[i][j+3] > 0 ? 1 : -1;
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

    int get_move() {
        return this->move;
    }
  /*
    0000000
    0000000
    0000000
    0000000
    0000000
    0000000
  */
    std::string print_board() {
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

    void show_board() {
        printf("column detail: ");
        for (int i = 0 ; i < 7; ++i) {
            printf("%d ", col[i]);
        }
        printf("\n");
        auto s = print_board();
        std::cout << std::endl;
        for (auto &ch : s) {
            if (ch == 'X') {
                 printf("X");
            } else if (ch == 'O') {
                printf("O");
            } else {
                printf("%c", ch);
            }
        }
        
        std::cout << std::endl;
    }

private:
    int a[6][7], col[7], move = 0;
};